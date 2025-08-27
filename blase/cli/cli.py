import argparse
import shutil
from pathlib import Path
import importlib.resources as pkg_resources

import blase.cli.templates
from blase.cli.restore.restore_cli import cmd_list, cmd_show, cmd_plan, cmd_run

TEMPLATE_CHOICES = ["pipeline", "standard", "dummy"] # List of template sub dirs

def create_project(template_type, project_name):
    if template_type not in TEMPLATE_CHOICES:
        print(f"Error: Template '{template_type}' not found. Available options: {TEMPLATE_CHOICES}")
        return

    dest_path = Path(project_name)
    if dest_path.exists():
        print(f"Directory '{project_name}' already exists.")
        return

    # Get the correct template folder inside templates/
    template_path = pkg_resources.files(blase.cli.templates) / template_type

    # Copy the chosen template
    shutil.copytree(template_path, dest_path)
    print(f"Project '{project_name}' created using template '{template_type}' at: {dest_path.resolve()}")

def main():
    """
    Entry point for the ``blase`` command-line interface (CLI).

    This function sets up the top-level CLI parser, defines all subcommands,
    and dispatches execution to the appropriate backend function.

    Supported commands
    ------------------
    **Project creation**

    * ``blase create <template> <name>``  
      Create a new project directory from a built-in template.

    **Restore/inspection**

    * ``blase restore list [--run <RUN>] [--like-fqn <FQN>] [--limit N]``  
      List recent steps recorded in a run.

    * ``blase restore show --step <STEP_HASH> [--run <RUN>]``  
      Show details for a specific step.

    * ``blase restore plan --step <STEP_HASH> [--run <RUN>]``  
      Display the restore plan for a step, including input/output availability.

    * ``blase restore run --step <STEP_HASH> [options]``  
      Execute restore by step. Modes include:
        - ``verify``: stream/inspect output without writing
        - ``materialize``: attempt to materialize outputs without replay
        - ``replay``: re-execute the step to reproduce outputs

    * ``blase restore run --data <DATA_HASH> [options]``  
      Execute restore by data hash. Modes include:
        - ``materialize``: attempt to retrieve the artifact from CAS/materializations
        - ``replay``: re-execute the producer step if replay is required

    Parameters
    ----------
    None

    Returns
    -------
    None
        Executes the requested CLI command. May call ``SystemExit`` for usage errors.

    Notes
    -----
    Shared options for ``restore run`` include:

    * ``--mode {verify, materialize, replay}``
    * ``--to <PATH>``: destination for outputs
    * ``--on-conflict {fail, rename, overwrite}``
    * ``--keep-intermediates``: retain temporary files during replay
    * ``--limit-batches N``: cap number of batches in verify mode
    * ``--backend {pandas, polars}``: override sink replay backend
    * ``--step`` vs ``--data``: mutually exclusive targets

    Example usage
    -------------
    Create a project::

        blase create standard my_project

    Inspect a step::

        blase restore show --step deadbeef...

    Plan a restore::

        blase restore plan --step cafebabe...

    Materialize by data hash::

        blase restore run --data abc123... --mode materialize --to out.csv
    """
    parser = argparse.ArgumentParser(prog="blase")
    subparsers = parser.add_subparsers(dest="command")

    # `blase create <template> <name>`
    create_parser = subparsers.add_parser("create", help="Create a new project")
    create_parser.add_argument("template", choices=TEMPLATE_CHOICES, help="Type of project template")
    create_parser.add_argument("name", help="Project directory name")

    # --- Restore related commands ---
    
    rst = subparsers.add_parser("restore", help="Restore/inspect recorded steps and data")
    rst_sub = rst.add_subparsers(dest="sub")

    ls = rst_sub.add_parser("list", help="List recent steps")
    ls.add_argument("--run", help="Run id or path")
    ls.add_argument("--like-fqn")
    ls.add_argument("--limit", type=int, default=20)
    ls.set_defaults(func=cmd_list)

    sh = rst_sub.add_parser("show", help="Show a step's details")
    sh.add_argument("--run")
    sh.add_argument("--step", required=True)
    sh.set_defaults(func=cmd_show)

    pl = rst_sub.add_parser("plan", help="Plan restore (what exists vs replay)")
    pl.add_argument("--run")
    pl.add_argument("--step", required=True)
    pl.set_defaults(func=cmd_plan)

    rn = rst_sub.add_parser("run", help="Execute restore")

    # Shared options
    rn.add_argument("--run")
    rn.add_argument("--mode", choices=["verify", "materialize", "replay"], default="verify")
    rn.add_argument("--to", help="Destination path or directory")
    rn.add_argument("--on-conflict", choices=["fail", "rename", "overwrite"])
    rn.add_argument("--keep-intermediates", action="store_true",
                help="Keep intermediate materializations created during replay.")
    rn.add_argument("--limit-batches", type=int)
    rn.add_argument("--backend", choices=["pandas", "polars"],
                    help="Override backend when replaying Load.save_to_csv")

    # Mutually exclusive: exactly one of --step or --data
    target_group = rn.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--step", help="Restore by step hash")
    target_group.add_argument("--data", help="Restore by data hash")

    rn.set_defaults(func=cmd_run)

    args = parser.parse_args()
    if args.command == "create":
        create_project(args.template, args.name)
        return
    if args.command == "restore" and hasattr(args, "func"):
        return args.func(args)
    parser.print_help()

if __name__ == "__main__":
    main()
