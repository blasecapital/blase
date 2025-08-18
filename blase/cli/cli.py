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
    Command:
    blase create <template_subdir_name> <project_name>

    blase restore list --limit 10
    blase restore show --step <STEP_HASH>
    blase restore plan --step <STEP_HASH>

    blase restore run --step <STEP_HASH> --mode verify
    blase restore run --step <STEP_HASH> --mode replay
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

    l = rst_sub.add_parser("list", help="List recent steps")
    l.add_argument("--run", help="Run id or path")
    l.add_argument("--like-fqn")
    l.add_argument("--limit", type=int, default=20)
    l.set_defaults(func=cmd_list)

    sh = rst_sub.add_parser("show", help="Show a step's details")
    sh.add_argument("--run")
    sh.add_argument("--step", required=True)
    sh.set_defaults(func=cmd_show)

    pl = rst_sub.add_parser("plan", help="Plan restore (what exists vs replay)")
    pl.add_argument("--run")
    pl.add_argument("--step", required=True)
    pl.set_defaults(func=cmd_plan)

    rn = rst_sub.add_parser("run", help="Execute restore")
    rn.add_argument("--run")
    rn.add_argument("--step", required=True)
    rn.add_argument("--mode", choices=["verify", "materialize", "replay"], default="verify")
    rn.add_argument("--to", help="Destination path or directory")
    rn.add_argument("--on-conflict", choices=["fail", "rename", "overwrite"])
    rn.add_argument("--limit-batches", type=int)
    rn.add_argument("--backend", choices=["pandas", "polars"], help="Override backend when replaying Load.save_to_csv")
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
