"""默认训练方案：1,200 步 × 32 条序列 × 每条 256 个目标。"""
import argparse  # 解析命令行参数
import json  # 读取配置和保存训练指标
import math  # 计算学习率计划
from pathlib import Path  # 处理配置和输出路径
import time  # 统计训练耗时
import torch  # 训练张量、随机采样和优化器
from torch.nn import functional as F  # 交叉熵损失
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha  # 公共工具
from evaluate import score  # 训练过程中计算 validation 分数


def main():
    # 记录整个运行过程的总时间。
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)

    # 指定使用哪个模型模块：model 或 student。
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()

    # 防止新实验覆盖旧结果。
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')

    # 设置设备、精度和线程数。
    device, precision = setup(args.device, args.precision, args.threads)

    # 固定随机种子，方便复现实验。
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()

    # 加载并校验训练、验证和测试数据。
    data = load_data()
    config = json.loads(args.config.read_text())

    # 创建模型，并检查固定的 context 和词表大小。
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    # 使用 AdamW 更新模型参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    tokens = data['train'][0].to(device)

    # 控制训练样本的随机起点。
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    for step in range(args.steps):
        # 每条样本取 257 个 token，用于构造 256 个输入-目标对。
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]

        # 前期 warmup，之后使用 cosine 衰减学习率。
        learning_rate = .001 * min(1.,(step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate

        # 输入和目标错开一位，计算 next-token loss。
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())

        # 反向传播并更新一次参数。
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            # 定期记录 validation 曲线，不使用 test 选择模型。
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds

    # 保存模型前进行最终 validation。
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'

    # 保存评估所需的模型信息和权重。
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256},checkpoint)

    # 保存实验指标和哈希，方便复现。
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
