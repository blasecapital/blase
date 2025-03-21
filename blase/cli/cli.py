import argparse
import shutil
from pathlib import Path
import importlib.resources as pkg_resources
import blase.cli.templates

TEMPLATE_CHOICES = ["pipeline"] # List of template sub dirs

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
    """
    parser = argparse.ArgumentParser(prog="blase")
    subparsers = parser.add_subparsers(dest="command")

    # `blase create <template> <name>`
    create_parser = subparsers.add_parser("create", help="Create a new project")
    create_parser.add_argument("template", choices=TEMPLATE_CHOICES, help="Type of project template")
    create_parser.add_argument("name", help="Project directory name")

    args = parser.parse_args()

    if args.command == "create":
        create_project(args.template, args.name)

if __name__ == "__main__":
    main()
