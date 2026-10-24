import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

try:
    from thop import clever_format
except:
    pass
    # you cannot use cal_param without profile


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024.0 / 1024.0
    return 0.0


def get_gpu_max_memory_mb():
    """Get peak GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
    return 0.0


def reset_gpu_memory_stats():
    """Reset GPU memory statistics"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()
        
    if getattr(args, 'benchmark', False):
        # Reset GPU memory stats before benchmarking
        reset_gpu_memory_stats()
        infer_time_meter = common_utils.AverageMeter()
        warmup_iterations = 10  # Warmup iterations to stabilize GPU
        benchmark_iterations = 100  # Number of iterations to benchmark
        min_time = float('inf')
        max_time = 0.0
        
    if getattr(args, 'cal_params', False):
        flops_meter = common_utils.AverageMeter()
        acts_meter = common_utils.AverageMeter()
        
    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        # Benchmark mode: measure inference time and memory
        if getattr(args, 'benchmark', False):
            if i < warmup_iterations:
                # Warmup phase
                with torch.no_grad():
                    _ = model(batch_dict)
                torch.cuda.synchronize()  # Wait for GPU to finish
                continue
            elif i >= warmup_iterations + benchmark_iterations:
                # Stop after benchmark iterations
                break
            
            # Synchronize before timing
            torch.cuda.synchronize()
            inference_start = time.time()
            
            with torch.no_grad():
                pred_dicts, ret_dict = model(batch_dict)
            
            # Synchronize after inference
            torch.cuda.synchronize()
            inference_time = time.time() - inference_start
            
            inference_time_ms = inference_time * 1000  # Convert to ms
            infer_time_meter.update(inference_time_ms)
            min_time = min(min_time, inference_time_ms)
            max_time = max(max_time, inference_time_ms)
            current_memory = get_gpu_memory_mb()
            
            disp_dict = {
                'infer_time_ms': f'{inference_time_ms:.2f}',
                'avg_time_ms': f'{infer_time_meter.avg:.2f}',
                'mem_mb': f'{current_memory:.1f}'
            }
        else:
            with torch.no_grad():
                pred_dicts, ret_dict = model(batch_dict)

            disp_dict = {}

            if getattr(args, 'infer_time', False):
                inference_time = time.time() - start_time
                infer_time_meter.update(inference_time * 1000)
                # use ms to measure inference time
                disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        if getattr(args, 'cal_params', False):
            macs, params, acts = common_utils.cal_flops(model, batch_dict)
            flops_meter.update(macs)
            acts_meter.update(acts)

        # Skip evaluation-related operations in benchmark mode
        if not getattr(args, 'benchmark', False):
            statistics_info(cfg, ret_dict, metric, disp_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=final_output_dir if args.save_to_file else None
            )
            det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if getattr(args, 'cal_params', False):
        macs, params, acts = clever_format([flops_meter.avg, params, acts_meter.avg], "%.3f")
        print(f'\nparams: {params}\nmacs: {macs}\nacts: {acts}\n')
        logger.info('params: %s \n' % params)
        logger.info('macs: %s \n' % macs)
        logger.info('acts: %s \n' % acts)
        
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if getattr(args, 'benchmark', False):
        max_memory_mb = get_gpu_max_memory_mb()
        if infer_time_meter.count > 0:
            logger.info('=' * 60)
            logger.info('*************** BENCHMARK RESULTS *****************')
            logger.info(f'Batch Size: {args.batch_size}')
            logger.info(f'Warmup Iterations: {warmup_iterations}')
            logger.info(f'Benchmark Iterations: {infer_time_meter.count}')
            logger.info(f'Average Inference Time per Frame: {infer_time_meter.avg:.2f} ms')
            if min_time != float('inf'):
                logger.info(f'Min Inference Time: {min_time:.2f} ms')
                logger.info(f'Max Inference Time: {max_time:.2f} ms')
            logger.info(f'Peak GPU Memory Usage: {max_memory_mb:.2f} MB ({max_memory_mb/1024:.2f} GB)')
            logger.info(f'FPS (Frames Per Second): {1000.0 / infer_time_meter.avg:.2f}')
            logger.info('=' * 60)
            print('\n' + '=' * 60)
            print('*************** BENCHMARK RESULTS *****************')
            print(f'Batch Size: {args.batch_size}')
            print(f'Average Inference Time per Frame: {infer_time_meter.avg:.2f} ms')
            if min_time != float('inf'):
                print(f'Min Inference Time: {min_time:.2f} ms')
                print(f'Max Inference Time: {max_time:.2f} ms')
            print(f'Peak GPU Memory Usage: {max_memory_mb:.2f} MB ({max_memory_mb/1024:.2f} GB)')
            print(f'FPS (Frames Per Second): {1000.0 / infer_time_meter.avg:.2f}')
            print('=' * 60 + '\n')
        else:
            logger.warning('No benchmark iterations completed!')
            print('Warning: No benchmark iterations completed!')
        
    if getattr(args, 'infer_time', False):
        print(model.module_time_meter)
        if hasattr(model.dense_head, 'time_meter'):
            print(model.dense_head.time_meter)
    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    # Skip evaluation if in benchmark mode
    if getattr(args, 'benchmark', False):
        return {}
    
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)
    
    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    dist_ranges = cfg.MODEL.POST_PROCESSING.get('DIST_RANGES', [None])
    if not isinstance(dist_ranges, list):
        dist_ranges = [dist_ranges]

    for dist_range in dist_ranges:
        if dist_range is not None:
            logger.info(f'*************** Distance Range: {dist_range} *****************')
        
        result_str, result_dict = dataset.evaluation(
            det_annos, class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=final_output_dir,
            dist_range=dist_range
        )

        logger.info(result_str)
        ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
