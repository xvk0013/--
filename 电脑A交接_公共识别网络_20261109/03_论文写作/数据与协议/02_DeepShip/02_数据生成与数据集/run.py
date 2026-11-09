"""Standalone DeepShip Python generator: plan, smoke, then full generation."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_name,'1')
import argparse
from collections import Counter
from pathlib import Path
import time
from common import SPLITS,load_config,read_json,write_json,read_csv,write_csv,safe_output
from dataset import set_schema
from planner import make_plan
from render import component_key,render_components,render_mixtures,prepare_background_parameters

HERE=Path(__file__).resolve().parent


def semantic_config(cfg):
    keys=('version','raw_root','classes','sample_rate','duration_seconds','channel','seed','output_rms',
          'sources','selection','bellhop','mixture','background','single_augmentation','generation_mode')
    return {key:cfg[key] for key in keys}


def plan_summary(rows):
    result={}
    for split in SPLITS:
        group=[row for row in rows if row['split']==split]
        singles=[row for row in group if row['kind']=='single']
        doubles=[row for row in group if row['kind']=='double']
        result[split]=dict(n=len(group),kind_counts=dict(Counter(row['kind'] for row in group)),
            single_per_class=dict(Counter(row['source1_class_name'] for row in singles)),
            double_per_pair=dict(Counter(row.get('label_name',str(row['label_index'])) for row in doubles)),
            double_context_counts=dict(Counter(row['context_condition'] for row in doubles)),
            source_recordings=len({row['source1_recording_id'] for row in singles}))
    return result


def report(directory,cfg,summary):
    lines=['# DeepShip生成结果','',f"状态：{summary['status']}；版本：{cfg['version']}；模式：{cfg['generation_mode']}。",'',
        '| 集合 | Single | Double | 合成背景 | 总数 | 单源录音数 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for split,info in summary['plan'].items():
        counts=info['kind_counts']
        lines.append(f"| {split} | {counts.get('single',0)} | {counts.get('double',0)} | {counts.get('background',0)} | {info['n']} | {info['source_recordings']} |")
    lines+=['',f"原录音归属沿用筛选索引；源片段每录音/事件上限为{cfg['selection']['max_per_recording']}，每类各集合配额为{cfg['selection']['per_class']}。preserve_existing_selected={cfg['selection']['preserve_existing_selected']}，开启时保留初筛已经选入的来源，再补到目标数。smoke仅用于小规模贯通。",'',
        f"信道：{'已有Bellhop E99 FIR' if cfg['bellhop']['enabled'] else '无额外Bellhop传播对照'}。使用真实核心前史/后视，不给每个5秒边界人为补零。",'',
        f"Double在接收组件、包络之前设置SIR={cfg['mixture']['sir_db']}dB；C0全重叠、C1单侧占优、C2交替占优，三个集合使用同一包络参数分布。最后施加公共增益至RMS={cfg['output_rms']}。",'',
        '背景为Wenz三成分合成负样本。Single基础数组不加人工包络，Train加载器可显式启用包络增强。原船声保留天然背景；后续SNR指相对已有接收波形的追加噪声比。', '',
        'DATA_INDEX.csv为最终唯一样本排列索引；data/Train、Val、Test/audio.npy为float32波形。components/audio.npy用于复用单源传播，STATE.json保存组件映射与物理处理信息。', '',
        '## 类别与组合','', '| 集合 | Single各类 | Double各组合 | C0/C1/C2 |','| --- | --- | --- | --- |']
    for split,info in summary['plan'].items():
        lines.append('| '+split+' | '+str(info['single_per_class'])+' | '+str(info['double_per_pair'])+' | '+str(info['double_context_counts'])+' |')
    if summary.get('rendered'):
        lines+=['','## 波形输出','', '| 集合 | RMS最小 | RMS最大 | 绝对峰值最大 |','| --- | ---: | ---: | ---: |']
        for split,info in summary['rendered'].items():
            lines.append(f"| {split} | {info['rms_min']:.8f} | {info['rms_max']:.8f} | {info['maximum_absolute_sample']:.8f} |")
    lines+=['',f"耗时：{summary.get('elapsed_seconds',0)/60:.2f}分钟。{summary.get('note','')}",'']
    (Path(directory)/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def run(args):
    cfg=load_config(args.config)
    cfg['generation_mode']='smoke' if args.mode=='smoke' else 'full'
    if args.mode=='smoke':
        size=int(cfg['runtime']['smoke_sources_per_class_split'])
        cfg['selection']['per_class']={split:size for split in SPLITS}
        cfg['selection']['preserve_existing_selected']=False
    cfg['set_names']=list(set_schema(cfg['classes'])[0])
    default_output=cfg['smoke_output_directory'] if args.mode=='smoke' else cfg['output_directory']
    output=args.output or default_output
    directory=safe_output(output,cfg['raw_root'])
    config_path=directory/'CONFIG.json';index_path=directory/'DATA_INDEX.csv';summary_path=directory/'SUMMARY.json'
    if config_path.exists() and semantic_config(read_json(config_path))!=semantic_config(cfg) and not args.overwrite_plan:
        raise ValueError('This output directory belongs to a different generation protocol. Choose a new output directory or explicitly use --overwrite-plan.')
    if index_path.exists() and not args.overwrite_plan:
        rows=read_csv(index_path)
    else:
        rows=make_plan(cfg)
        prepare_background_parameters(cfg,rows)
        write_csv(index_path,rows)
    plan=plan_summary(rows);begun=time.perf_counter()
    for split,info in plan.items():
        print(f"PLAN {split}: {info['kind_counts']} total={info['n']} source_recordings={info['source_recordings']}",flush=True)
    n=cfg['sample_rate']*cfg['duration_seconds']
    total=sum(info['n'] for info in plan.values());sources=sum(row['kind']=='single' for row in rows)
    print(f"Planned wave arrays={total*n*4/2**30:.3f}GiB; receiving components={sources*n*4/2**30:.3f}GiB",flush=True)
    if summary_path.exists() and not args.overwrite_plan and not args.mix_only and not args.overwrite_components:
        previous=read_json(summary_path)
        if previous.get('status')=='complete':
            print('This dataset is already complete; no regeneration.',flush=True)
            print('Dataset:',directory,flush=True)
            return
    write_json(config_path,cfg)
    summary=dict(status='planned',version=cfg['version'],plan=plan,elapsed_seconds=0)
    write_json(summary_path,summary);report(directory,cfg,summary)
    if args.mode=='plan':
        print('Plan saved. No waveform generated. Index:',index_path,flush=True)
        return
    try:
        write_json(summary_path,dict(summary,status='running'))
        if args.mix_only:
            state=read_json(directory/'components'/'STATE.json')
            if state.get('status')!='complete' or state.get('key')!=component_key(cfg,rows):
                raise ValueError('Reusable receiving components do not match this plan.')
        else:
            state=render_components(cfg,directory,rows,overwrite=args.overwrite_components)
        if args.components_only:
            summary.update(status='components_complete',elapsed_seconds=time.perf_counter()-begun,
                note='单源组件已完成，最终识别波形尚未混合生成。')
            write_json(summary_path,summary);report(directory,cfg,summary)
            print('Receiving components complete:',directory/'components'/'audio.npy',flush=True)
            return
        rendered=render_mixtures(cfg,directory,rows,state)
        write_csv(index_path,rows)
        summary.update(status='complete',elapsed_seconds=time.perf_counter()-begun,rendered=rendered)
        write_json(summary_path,summary);report(directory,cfg,summary)
        print('COMPLETE:',directory,flush=True)
        print('Report:',directory/'REPORT.md',flush=True)
        print('Index:',index_path,flush=True)
    except BaseException as error:
        summary.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
            elapsed_seconds=time.perf_counter()-begun,error=str(error))
        write_json(summary_path,summary)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(HERE/'config.json'))
    parser.add_argument('--mode',choices=('plan','smoke','full'),default='smoke')
    parser.add_argument('--output',help='Override this dataset output directory (must be outside raw_root).')
    parser.add_argument('--components-only',action='store_true',help='Generate only the reusable single-source receiving components.')
    parser.add_argument('--mix-only',action='store_true',help='Reuse existing receiving components and render the mixture arrays.')
    parser.add_argument('--overwrite-plan',action='store_true',help='Explicitly replace this output source/mixing plan; original data is never deleted.')
    parser.add_argument('--overwrite-components',action='store_true',help='Explicitly rebuild this output receiving-component array.')
    args=parser.parse_args()
    if args.components_only and args.mix_only:
        parser.error('--components-only and --mix-only cannot be combined.')
    run(args)


if __name__=='__main__':
    main()
