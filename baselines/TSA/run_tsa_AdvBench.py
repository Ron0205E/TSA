import sys
import os
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

current_dir = Path(__file__).resolve().parent
baselines_dir = current_dir.parent
project_root = baselines_dir.parent
sys.path.insert(0, str(project_root))

from TSA_AdvBench import TSA

class ThreadSafeTSAOutputFormatter:

    def __init__(
            self,
            cutoff_score: int = 9,
            live_path: str = None,
            write_all_live: bool = True,
            snapshot_path: str = None,
            save_all_to_json: bool = True,
            auto_resume: bool = True
    ):
        self.cutoff_score = cutoff_score
        self.results = []  # Successful results
        self.all_results = []  # All results
        self.live_path = live_path
        self.write_all_live = write_all_live
        self.snapshot_path = snapshot_path
        self.save_all_to_json = save_all_to_json

        # Thread locks
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()

        # Progress statistics
        self.processed_count = 0
        self.success_count = 0
        self.failed_count = 0

        # Set of processed behavior_ids (for resume filtering)
        self.processed_ids: Set[str] = set()

        # Ensure output directory exists
        if live_path:
            os.makedirs(os.path.dirname(live_path), exist_ok=True)
        if snapshot_path:
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)

        # Auto-resume mode: always try to load existing results
        if auto_resume and snapshot_path and Path(snapshot_path).exists():
            self._load_existing_results(snapshot_path)

        # Initialize live file (if it doesn't exist)
        if live_path and not os.path.exists(live_path):
            with open(live_path, 'w', encoding='utf-8') as f:
                f.write('')

    def _load_existing_results(self, snapshot_path: str):
        """Load existing result file (resume mode)"""
        try:
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.all_results = data.get('results', [])

            # Rebuild processed_ids set
            for r in self.all_results:
                behavior_id = r.get('behavior_id')
                if behavior_id:
                    self.processed_ids.add(behavior_id)
                if r.get('is_jailbroken'):
                    self.results.append(r)

            self.processed_count = len(self.all_results)
            self.success_count = len(self.results)
            self.failed_count = self.processed_count - self.success_count

            print(f"\n📂 [Auto-resume] Loaded {self.processed_count} historical results")
            print(f"             Successful: {self.success_count}, Failed: {self.failed_count}")

        except Exception as e:
            print(f"⚠️ Failed to load historical results, starting from scratch: {e}")
            self.all_results = []
            self.results = []
            self.processed_ids = set()

    def add_result(
            self,
            behavior_id: str,
            behavior: str,
            context_str: str,
            attack_prompt: str,
            target_response: str,
            score: int,
            stage1_prompt: str = '',
            stage1_response: str = '',
            stage2_prompt: str = '',
            stage2_response: str = ''
    ) -> Dict:
        """Add a result and immediately save to file (thread-safe)"""
        is_jailbroken = score >= self.cutoff_score

        result = {
            'behavior_id': behavior_id,
            'behavior': behavior,
            'context_str': context_str,
            'attack_prompt': attack_prompt,
            'target_response': target_response,
            'stage1_prompt': stage1_prompt,
            'stage1_response': stage1_response,
            'stage2_prompt': stage2_prompt,
            'stage2_response': stage2_response,
            'score': score,
            'is_jailbroken': is_jailbroken
        }

        # Use lock to protect shared data
        with self._lock:
            # Check if already exists to avoid duplicates
            exists = False
            for i, r in enumerate(self.all_results):
                if r.get('behavior_id') == behavior_id:
                    self.all_results[i] = result
                    exists = True
                    break
            if not exists:
                self.all_results.append(result)
                self.processed_ids.add(behavior_id)
                self.processed_count += 1

            # Update successful cases list
            if is_jailbroken:
                # Remove old entry if it exists
                self.results = [r for r in self.results if r.get('behavior_id') != behavior_id]
                self.results.append(result)
                self.success_count = len(self.results)
            else:
                # If previously successful but now failed, remove from success list
                self.results = [r for r in self.results if r.get('behavior_id') != behavior_id]
                self.success_count = len(self.results)

            self.failed_count = self.processed_count - self.success_count

        # Write to JSONL in real-time (using file lock)
        if self.live_path and (self.write_all_live or is_jailbroken):
            with self._file_lock:
                with open(self.live_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()

        # Overwrite snapshot (update every 5 entries or on success)
        if self.snapshot_path and (is_jailbroken or self.processed_count % 5 == 0):
            self._write_snapshot()

        return result

    def _write_snapshot(self):
        """Write snapshot file (thread-safe)"""
        with self._file_lock:
            with self._lock:
                snapshot_data = {
                    'method': 'TSA (Two-Stage Attack) - Multi-threaded',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_processed': self.processed_count,
                    'total_jailbroken': self.success_count,
                    'cutoff_score': self.cutoff_score,
                    'results': self.all_results if self.save_all_to_json else self.results
                }
            with open(self.snapshot_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot_data, f, ensure_ascii=False, indent=2)

    def save_to_json(self, output_path: str):
        """Save final results to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with self._lock:
            output_data = {
                'method': 'TSA (Two-Stage Attack) - Multi-threaded',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_processed': self.processed_count,
                'total_jailbroken': self.success_count,
                'cutoff_score': self.cutoff_score,
                'results': self.all_results if self.save_all_to_json else self.results
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def get_summary(self) -> Dict:
        """Get result summary (thread-safe)"""
        with self._lock:
            return {
                'total_processed': self.processed_count,
                'total_jailbroken': self.success_count,
                'cutoff_score': self.cutoff_score,
                'success_count': self.success_count,
                'failed_count': self.failed_count
            }

    def update_progress(self, current: int, total: int):
        """Update progress (for display)"""
        with self._lock:
            print(
                f"\r[Progress] {current}/{total} | Success: {self.success_count} | Failed: {self.failed_count}",
                end='', flush=True)

    def is_processed(self, behavior_id: str) -> bool:
        """Check if a behavior has already been processed (for resume filtering)"""
        return behavior_id in self.processed_ids


def parse_indices(indices_str: str) -> List[int]:
    """Parse index string, supports formats: 1,2,3 or 1-5 or 1,3-5,7"""
    indices = []
    for part in indices_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def load_behaviors(data_path: str, dataset_format: str = 'advbench') -> List[Dict[str, str]]:
    """Load behavior data from JSON file"""
    data_path = Path(data_path)

    if not data_path.exists():
        possible_path = project_root / data_path
        if possible_path.exists():
            data_path = possible_path
        else:
            raise FileNotFoundError(f"Behavior data file not found: {data_path}")

    print(f"Reading data file: {data_path}")

    behaviors = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            for idx, item in enumerate(data):
                behaviors.append({
                    'BehaviorID': item.get('BehaviorID', f'advbench_{idx}'),
                    'Behavior': item.get('prompt', item.get('Behavior', '')),
                    'ContextString': item.get('target', item.get('ContextString', ''))
                })
        elif isinstance(data, dict) and 'data' in data:
            for idx, item in enumerate(data['data']):
                behaviors.append({
                    'BehaviorID': item.get('BehaviorID', f'advbench_{idx}'),
                    'Behavior': item.get('prompt', item.get('Behavior', '')),
                    'ContextString': item.get('target', item.get('ContextString', ''))
                })
        else:
            raise ValueError(f"Unsupported JSON format")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

    print(f"Successfully loaded {len(behaviors)} behavior entries")
    return behaviors


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML or JSON configuration file"""
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if str(config_path).endswith(('.yaml', '.yml')):
        try:
            import yaml
        except ImportError:
            raise ImportError("Please install PyYAML: pip install pyyaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif str(config_path).endswith('.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format")


def process_single_behavior(
        tsa_instance: TSA,
        behavior: Dict,
        orig_idx: int,
        formatter: ThreadSafeTSAOutputFormatter,
        verbose: bool = True
) -> Dict:
    """Process a single behavior - with two-stage progress display"""
    behavior_id = behavior.get('BehaviorID', f'behavior_{orig_idx}')
    thread_name = threading.current_thread().name

    try:
        if verbose:
            print(f"\n[Thread {thread_name}] ⏳ Starting to process behavior [{orig_idx}]: {behavior_id}")
            print(f"[Thread {thread_name}]   Behavior content: {behavior.get('Behavior', '')[:100]}...")

        # Stage 1: Generate analysis report
        if verbose:
            print(f"[Thread {thread_name}]   📝 Stage 1/2: Generating analysis report...")

        test_case, logs, jailbreak_info = tsa_instance.generate_test_cases_single_behavior(
            behavior, verbose=False  # Don't print detailed logs inside thread, we control it ourselves
        )

        if verbose:
            print(f"[Thread {thread_name}]   ✅ Stage 1/2: Complete")
            print(f"[Thread {thread_name}]   🔍 Stage 2/2: Extracting operational steps...")

        if verbose:
            print(f"[Thread {thread_name}]   ✅ Stage 2/2: Complete")

        # Add result to formatter
        formatter.add_result(
            behavior_id=jailbreak_info['behavior_id'],
            behavior=jailbreak_info['behavior'],
            context_str=jailbreak_info['context_str'],
            attack_prompt=jailbreak_info['attack_prompt'],
            target_response=jailbreak_info['target_response'],
            score=jailbreak_info['score'],
            stage1_prompt=jailbreak_info.get('stage1_prompt', ''),
            stage1_response=jailbreak_info.get('stage1_response', ''),
            stage2_prompt=jailbreak_info.get('stage2_prompt', ''),
            stage2_response=jailbreak_info.get('stage2_response', '')
        )

        if verbose:
            status = "✓ Jailbreak successful" if jailbreak_info['is_jailbroken'] else "✗ Jailbreak failed"
            status_icon = "✅" if jailbreak_info['is_jailbroken'] else "❌"
            print(f"[Thread {thread_name}]   {status_icon} Score: {jailbreak_info['score']}/10 | {status}")
            print(f"[Thread {thread_name}]   💾 Result saved")

        return {
            'orig_idx': orig_idx,
            'behavior_id': behavior_id,
            'success': True,
            'score': jailbreak_info['score'],
            'is_jailbroken': jailbreak_info['is_jailbroken']
        }
    except Exception as e:
        if verbose:
            print(f"\n[Thread {thread_name}] ❌ Failed to process behavior {behavior_id}: {e}")
            print(f"[Thread {thread_name}]    Error details: {str(e)}")

        formatter.add_result(
            behavior_id=behavior_id,
            behavior=behavior.get('Behavior', ''),
            context_str=behavior.get('ContextString', ''),
            attack_prompt='',
            target_response=f"ERROR: {e}",
            score=1
        )

        return {
            'orig_idx': orig_idx,
            'behavior_id': behavior_id,
            'success': False,
            'error': str(e)
        }

def run_tsa_attack_multi_thread(
        config_path: str,
        behaviors_path: str,
        output_path: str,
        dataset_format: str = 'harmbench',
        num_behaviors: int = None,
        start_index: int = 0,
        end_index: int = None,
        indices: str = None,
        rerun_failed: bool = False,
        rerun_low_score: int = None,
        num_threads: int = 20,
        verbose: bool = True
) -> None:
    """Run TSA two-stage attack (auto-resume)"""
    print("=" * 60)
    print("        TSA (Two-Stage Attack) Attack Framework - Multi-threaded")
    print("=" * 60)
    print(f"Number of threads: {num_threads}")
    print("🔄 Auto-resume mode enabled (will automatically continue from breakpoint)")

    # 1. Load configuration
    print(f"\n[1/5] Loading configuration file: {config_path}")
    config = load_config(config_path)
    print("  ✓ Configuration loaded")

    # 2. Load behavior data
    print(f"\n[2/5] Loading behavior data: {behaviors_path}")
    all_behaviors = load_behaviors(behaviors_path, dataset_format=dataset_format)
    if num_behaviors:
        all_behaviors = all_behaviors[:num_behaviors]
    print(f"  ✓ Loaded {len(all_behaviors)} behaviors")

    # 3. Initialize TSA framework
    print("\n[3/5] Initializing TSA framework...")
    try:
        # Parse configuration (keeping original logic unchanged)
        if 'tsa_params' in config:
            tsa_params = config.get('tsa_params', {})
            models_config = config.get('models', {})
            output_params = config.get('output_params', {})
            tsa_target_model = tsa_params.get('target_model') or models_config.get('target_model')
            tsa_judge_model = tsa_params.get('judge_model') or models_config.get('judge_model')
            tsa_attack_params = tsa_params.get('attack', {})
            tsa_output_params = tsa_params.get('output', {})
        else:
            tsa_target_model = config.get('target_model')
            tsa_judge_model = config.get('judge_model')
            tsa_attack_params = config.get('attack', {})
            tsa_output_params = config.get('output', {})
            output_params = {}

        tsa_instance = TSA(
            target_model_config=tsa_target_model,
            judge_model_config=tsa_judge_model,
            judge_max_n_tokens=tsa_judge_model.get('max_n_tokens', 10) if tsa_judge_model else 10,
            cutoff_score=tsa_attack_params.get('cutoff_score') or output_params.get('cutoff_score', 9),
            max_retries=tsa_attack_params.get('max_retries') or config.get('attack_params', {}).get('max_retries', 5),
            target_max_n_tokens=tsa_target_model.get('max_n_tokens', 500) if tsa_target_model else 500,
            temperature=tsa_target_model.get('temperature', 0.5) if tsa_target_model else 0.5,
        )

        # Output path configuration
        tsa_output_dir = current_dir / 'output'
        os.makedirs(tsa_output_dir, exist_ok=True)
        live_file = tsa_output_params.get('live_file', 'live_results.jsonl')
        output_file = tsa_output_params.get('output_file', output_path)
        live_path = tsa_output_dir / live_file
        snapshot_path = tsa_output_dir / output_file
        save_all = tsa_output_params.get('save_all_to_json', True)
        cutoff = tsa_attack_params.get('cutoff_score') or output_params.get('cutoff_score', 9)

        # Create formatter (auto-resume enabled by default)
        formatter = ThreadSafeTSAOutputFormatter(
            cutoff_score=cutoff,
            live_path=str(live_path),
            write_all_live=True,
            snapshot_path=str(snapshot_path),
            save_all_to_json=save_all,
            auto_resume=True  # Force enable auto-resume
        )

        print("  ✓ TSA framework initialized")
        print(f"    - Target model: {tsa_target_model.get('model_name_or_path')}")
        print(f"    - Output file: {output_file}")

    except Exception as e:
        print(f"  ✗ Initialization failed: {e}")
        raise

    # 4. Prepare data to process
    print("\n[4/5] Preparing data...")

    # Determine processing range
    total_behaviors = len(all_behaviors)
    if end_index is None:
        end_index = total_behaviors
    end_index = min(end_index, total_behaviors)

    # Build processing list
    if indices:
        target_indices = set(parse_indices(indices))
        all_targets = [(i, all_behaviors[i]) for i in range(start_index, end_index) if i in target_indices]
    else:
        all_targets = [(i, all_behaviors[i]) for i in range(start_index, end_index)]

    # Auto-resume: filter out already processed behaviors
    original_count = len(all_targets)
    all_targets = [(i, b) for i, b in all_targets
                   if not formatter.is_processed(b.get('BehaviorID', f'behavior_{i}'))]
    skipped_count = original_count - len(all_targets)

    if skipped_count > 0:
        print(f"  ⏭️  Auto-resume: Skipping {skipped_count} already processed behaviors")
    else:
        print(f"  ✨ No historical records detected, starting from scratch")

    # Handle rerun logic
    if rerun_failed or rerun_low_score is not None:
        # Reload all results for processing
        all_results_map = {r['behavior_id']: r for r in formatter.all_results}

        # Find behaviors that need to be rerun
        rerun_targets = []
        for i, b in all_targets:
            bid = b.get('BehaviorID', f'behavior_{i}')
            if bid in all_results_map:
                r = all_results_map[bid]
                if (rerun_failed and not r.get('is_jailbroken', False)) or \
                        (rerun_low_score is not None and r.get('score', 0) < rerun_low_score):
                    rerun_targets.append((i, b))
                    # Remove from processed set to allow rerun
                    if bid in formatter.processed_ids:
                        formatter.processed_ids.remove(bid)
            else:
                rerun_targets.append((i, b))

        behaviors_to_process = rerun_targets
        print(f"  🔄 Rerun mode: Need to rerun {len(behaviors_to_process)} behaviors")
    else:
        behaviors_to_process = all_targets

    total_count = len(behaviors_to_process)
    print(f"  - To process: {total_count} records")
    print(f"  - Number of threads: {num_threads}")

    # 5. Run attack
    print("\n[5/5] Running TSA two-stage attack...")
    print(f"  Results saved to: {snapshot_path}")

    if total_count == 0:
        print("✅ All behaviors have been processed, no need to continue!")
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(process_single_behavior, tsa_instance, behavior, idx, formatter, verbose): (
                idx, behavior)
                for idx, behavior in behaviors_to_process
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()

                if completed % max(1, total_count // 20) == 0 or completed == total_count:
                    formatter.update_progress(completed, total_count)

    # 6. Save final results
    print("\n" + "-" * 60)
    final_output_path = tsa_output_dir / output_file
    print(f"\n[6/5] Saving results to: {final_output_path}")
    formatter.save_to_json(final_output_path)

    summary = formatter.get_summary()

    print("\n" + "=" * 60)
    print("          Attack Complete! Results Summary")
    print("=" * 60)
    print(f"Total behaviors: {summary['total_processed']}")
    print(f"Successful jailbreaks: {summary['total_jailbroken']}")
    if summary['total_processed'] > 0:
        print(f"Success rate: {summary['total_jailbroken'] / summary['total_processed'] * 100:.1f}%")
    print(f"Score threshold: {summary['cutoff_score']}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='TSA Attack Framework - Multi-threaded (Auto-resume + Two-stage Progress)')
    parser.add_argument('--config', type=str, default=None, help='Configuration file path')
    parser.add_argument('--output', type=str, default='tsa_results.json', help='Output file name')
    parser.add_argument('--num', type=int, default=None, help='Process only the first N behaviors')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, default=None, help='End index')
    parser.add_argument('--indices', type=str, default=None, help='Specify indices to process')
    parser.add_argument('--rerun-failed', action='store_true', help='Rerun failed records')
    parser.add_argument('--rerun-low-score', type=int, default=None, help='Rerun low score records')
    parser.add_argument('--threads', type=int, default=15, help='Number of threads')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    # Note: No --resume parameter, defaults to resume mode
    args = parser.parse_args()

    CONFIG_PATH = args.config if args.config else current_dir / "config_tsa.yaml"
    OUTPUT_PATH = args.output
    DATASET_FORMAT = 'miniAdvBench'
    BEHAVIORS_PATH = "dataset/miniAdvBench.json"


    run_tsa_attack_multi_thread(
        config_path=CONFIG_PATH,
        behaviors_path=project_root / BEHAVIORS_PATH,
        output_path=OUTPUT_PATH,
        dataset_format=DATASET_FORMAT,
        num_behaviors=args.num,
        start_index=args.start,
        end_index=args.end,
        indices=args.indices,
        rerun_failed=args.rerun_failed,
        rerun_low_score=args.rerun_low_score,
        num_threads=args.threads,
        verbose=not args.quiet
    )