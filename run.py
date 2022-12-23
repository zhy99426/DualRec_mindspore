import yaml, argparse
from tqdm import tqdm
import numpy as np
from src.datasets.base import AbstractDataset
from src.dataloaders.dualrec import DualRecDataloader
from src.models import DotProductPredictionHead
from src.models.dualrec import DualRecModel
from src.utils import recalls_and_ndcgs_for_ks, mrr

import mindspore as ms
from mindspore import nn, ops, Model, context

ms.set_context(mode=context.PYNATIVE_MODE)


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_yaml", type=str, default= "./src/config/beauty.yaml",help="YAML config file specifying default arguments")

def flatten_dict(cfg, flatten_cfg):
    for k,v in cfg.items():
        if isinstance(v, dict):
            flatten_dict(v, flatten_cfg)
        else:
            flatten_cfg[k] = v
            

configs = parser.parse_known_args()[0]
if configs.config_yaml:
    with open(configs.config_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        flatten_cfg = dict()
        flatten_dict(cfg, flatten_cfg)
        for k, v in flatten_cfg.items():
            parser.add_argument("--"+k, type=type(v))
        parser.set_defaults(**flatten_cfg)

args = parser.parse_args()

dataset = AbstractDataset(
            args.dataset_code
        )
dataloader = DualRecDataloader(
    dataset,
    args.data_type,
    args.seg_len,
    args.num_train_seg,
    args.num_test_seg,
    args.pred_prob,
    args.num_workers,
    args.test_negative_sampler_code,
    args.test_negative_sample_size,
    args.train_batch_size,
    args.val_batch_size,
    args.test_batch_size,
)

dualrec = DualRecModel(
    d_model=args.d_model,
    d_head=args.d_head,
    n_head=args.n_head,
    d_inner=args.d_inner,
    layer_norm_eps=args.layer_norm_eps,
    activation_type=args.activation_type,
    clamp_len=args.clamp_len,
    n_layer=args.n_layer,
    num_items=args.num_items,
    seg_len=args.seg_len,
    device=args.device,
    dropout=args.dropout,
    initializer_range=args.initializer_range,
    multi_scale=args.multi_scale,
    reverse=False,
)

head = DotProductPredictionHead(
    dualrec.d_model, dualrec.num_items, dualrec.item_embedding
)
Loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

dualrec_reversed = DualRecModel(
    d_model=args.d_model,
    d_head=args.d_head,
    n_head=args.n_head,
    d_inner=args.d_inner,
    layer_norm_eps=args.layer_norm_eps,
    activation_type=args.activation_type,
    clamp_len=args.clamp_len,
    n_layer=args.n_layer,
    num_items=args.num_items,
    seg_len=args.seg_len,
    device=args.device,
    dropout=args.dropout,
    initializer_range=args.initializer_range,
    multi_scale=args.multi_scale,
    reverse=True,
)
dualrec_reversed.dropout = dualrec.dropout
dualrec_reversed.item_embedding = dualrec.item_embedding



class TrainModel(nn.Cell):
    def __init__(self, dualrec, dualrec_reversed):
        super().__init__()
        self.dualrec = dualrec
        self.dualrec_reversed = dualrec_reversed
        
    def forward(self, input_ids1, input_ids2):
        outputs = []
        outputs_reversed = []
        for i in range(input_ids1.shape[1]):
            input_mask1 = ops.cast((input_ids1[:, i] == 0), ms.float32)
            input_mask2 = ops.cast((input_ids2[:, i] == 0), ms.float32)

            output, attn = self.dualrec(input_ids=input_ids1[:, i], input_mask=input_mask1, output_attentions=True)
            output_reversed, attn_reversed = self.dualrec_reversed(
                input_ids=input_ids2[:, i], input_mask=input_mask2, output_attentions=True
            )
            outputs.append(output)
            outputs_reversed.append(output_reversed)
        outputs = ops.Concat(axis=1)(outputs)
        outputs_reversed = ops.Concat(axis=1)(outputs_reversed)

        return outputs, outputs_reversed, attn, attn_reversed

    def eval(self, input_ids1):
        outputs = []
        for i in range(input_ids1.shape[1]):
            input_mask = ops.cast((input_ids1[:, i] == 0), ms.float32)
            output = self.dualrec(input_ids=input_ids1[:, i], input_mask=input_mask)
            outputs.append(output[0])
        outputs = ops.Concat(axis=1)(outputs)
        return outputs
    
    
    def construct(self, batch):
        input_ids1 = batch[:, :, :-1]
        input_ids2 = batch[:, :, 1:]
        
        outputs, outputs_reversed, attn, attn_reversed = self.forward(input_ids1, input_ids2)

        mask1 = ms.numpy.where(input_ids1.squeeze() != 0, 1, 0)
        mask2 = ms.numpy.where(input_ids2.squeeze() != 0, 1, 0)

        maskedselect = ops.MaskedSelect()
        
        mask1 = ops.cast(mask1, ms.bool_)
        mask2 = ops.cast(mask2, ms.bool_)

        logits1 = head(maskedselect(outputs, mask1.expand_dims(-1)).view(-1,64))  # BT x Hout
        loss1 = (Loss(logits1, maskedselect(input_ids2.squeeze(), mask1)).expand_dims(-1)).sum(axis=0)/logits1.shape[0]
        logits_reversed1 = head(maskedselect(outputs_reversed, mask2.expand_dims(-1)).view(-1,64))  # BT x H
        loss2 = (Loss(logits_reversed1, maskedselect(input_ids1.squeeze(), mask2)).expand_dims(-1)).sum(axis=0)/logits_reversed1.shape[0]

        loss = (args.loss_ratio * loss1 + (1-args.loss_ratio)*loss2)
        loss += args.aux_factor * (compute_kl_loss(attn[0].mean(axis=0).view(-1, self.dualrec.d_head), attn_reversed[0].mean(axis=0).view(-1, self.dualrec.d_head))+
                                    compute_kl_loss(attn[1].mean(axis=0).view(-1, self.dualrec.d_head), attn_reversed[1].mean(axis=0).view(-1, self.dualrec.d_head)))/2
        
        
        return loss
    
net_with_loss = TrainModel(dualrec, dualrec_reversed)
net_with_loss.to_float(ms.float32)
optimizer = nn.Adam(
        [{'params': net_with_loss.trainable_params()}], learning_rate=args.lr, weight_decay=float(args.weight_decay)
    )



def training_epoch_end(training_step_outputs):
    loss = ops.concat([o["loss"] for o in training_step_outputs], 0).mean()
    print("train_loss", loss)


def validation_step(model, batch):
    input_ids = batch["input_ids"]

    outputs = model(input_ids)

    # get scores (B x C) for evaluation
    last_outputs = outputs[:, -1, :]
    candidates = batch["candidates"].squeeze()  # B x C
    logits = head(last_outputs, candidates)  # B x C

    labels = batch["labels"].squeeze()
    metrics = recalls_and_ndcgs_for_ks(logits, labels, [1, 5, 10, 20, 50])
    metrics["MRR"] = mrr(logits, labels)
    return metrics

def validation_epoch_end(validation_step_outputs):
    metrics = dict()
    keys = validation_step_outputs[0].keys()
    for k in keys:
        tmp = []
        for o in validation_step_outputs:
            tmp.append(o[k])
        metrics[k] = ops.stack(tmp).mean()
        print("Val:"+k, metrics[k])
    return metrics

def test_epoch_end(test_step_outputs):
    metrics = dict()
    keys = test_step_outputs[0].keys()
    for k in keys:
        tmp = []
        for o in test_step_outputs:
            tmp.append(o[k])
        metrics[k] = ops.stack(tmp).mean()
        print("Test:"+k, metrics[k])
    return metrics

def compute_kl_loss(p, q, pad_mask=None):
    # p_loss = q * (ops.log_softmax(q, axis=-1) - ops.log_softmax(p, axis=-1))/p.shape[0]
    # q_loss = p * (ops.log_softmax(p, axis=-1) - ops.log_softmax(q, axis=-1))/q.shape[0]
    p_loss = ops.kl_div(
        ops.log_softmax(p, axis=-1), ops.softmax(q, axis=-1), reduction="none"
    )
    q_loss = ops.kl_div(
        ops.log_softmax(q, axis=-1), ops.softmax(p, axis=-1), reduction="none"
    )

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.0)
        q_loss.masked_fill_(pad_mask, 0.0)
    # # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = ms.numpy.sum(p_loss)/ p_loss.shape[0]
    q_loss = ms.numpy.sum(q_loss)/ q_loss.shape[0]

    loss = (p_loss + q_loss) / 2
    return loss
    
if __name__ == "__main__":
    model = nn.TrainOneStepCell(net_with_loss, optimizer=optimizer)
    best_mrr = 0
    early_stop_count = 0
    save_path = './dualrec_best_'+args.dataset_code+'.ckpt'
    for epoch in range(args.max_epochs):
        print("epoch: ", epoch)
        train_dataset = dataloader.get_train_loader()
        valid_dataset = dataloader.get_valid_loader()
        test_dataset = dataloader.get_test_loader()
        net_with_loss.set_train()
        for batch_idx, batch in tqdm(enumerate(train_dataset.create_dict_iterator())):
            loss = model(batch["input_ids"])
            
        if epoch % 5 == 0:
            net_with_loss.set_train(False)
            metrics = []
            for batch_idx, batch in tqdm(enumerate(valid_dataset.create_dict_iterator())):
                metrics.append(validation_step(net_with_loss.eval, batch))
            metrics = validation_epoch_end(metrics)
            if metrics["MRR"] > best_mrr:
                early_stop_count = 0
                best_mrr = metrics["MRR"]
                print("Saving Model ...")
                ms.save_checkpoint(dualrec, save_path)
            else:
                early_stop_count += 1
            if early_stop_count > 2:
                break
        
    print('Finished Training')
    metrics = []
    dualrec = ms.load_checkpoint(save_path)
    for batch_idx, batch in tqdm(enumerate(test_dataset.create_dict_iterator())):
        metrics.append(validation_step(net_with_loss.eval, batch))
    test_epoch_end(metrics)
