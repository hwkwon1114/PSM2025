#!/usr/bin/env python3
"""Run the supplied collaborator recipe without changing its loading parameters."""
import argparse
import json
import run_forward_model_diagnostics as runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=runner.IMPLEMENTATIONS, default="current")
    parser.add_argument("--max-iter", type=int, default=50000)
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()
    if cli.max_iter < 1:
        parser.error("--max-iter must be positive")
    name = "collaborator_crown" if cli.max_iter == 50000 else f"collaborator_crown_iter{cli.max_iter}"
    spec=dict(name=name,family='collaborator',res=.015,growth_per_hit=.0015,
              max_iter=cli.max_iter,
              note='Collaborator JSON; historical Putong-parity 254x304.8x0.6 mm plate, res=.015. Exact settings of collaborator crown run remain unconfirmed.',
              sequence=json.loads((runner.RUN_ROOT/'collaborator_crown_input.json').read_text()))
    args=argparse.Namespace(force=False,timeout=10800,threads=8,sequence_adaptive=False,
                            grad_tol=1e-11,seed_amplitude=.004,material_baseline=runner.MATERIAL_BASELINE)
    if cli.dry_run:
        binary = runner.IMPLEMENTATIONS[cli.implementation]["binary"]
        directory = runner.RUN_ROOT / cli.implementation / name
        print(json.dumps(runner.build_command(binary, spec, directory, cli.implementation, args), indent=2))
    else:
        raise SystemExit(runner.run_case(spec,cli.implementation,args,None))
