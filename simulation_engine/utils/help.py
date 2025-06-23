import argparse
from rich import print
from rich.table import Table
from rich.console import Console

class HelpPrintout(argparse.Action):
    def __init__(self, **kwargs):
        super().__init__(nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        self.print_settings_table()
        parser.exit()

    def print_settings_table(self):
        from simulation_engine.main.entrypoint import IMPLEMENTED_TYPES
        self.console = Console()
        long_line_divide = "[dim]" + "─" * 40 + "[/dim]"
        short_line_divide = "[dim]" + "─" * 20 + "[/dim]"
        table = Table(title="[bold]Simulation Engine Arguments ⚙️[/bold]")
        table.add_column("Setting", justify="centre",style="bold magenta")
        table.add_column("Command Flag", justify="left",style="yellow")
        table.add_column("Values", justify="left", style="green")

        table.add_row("⚙️  Mode", "(argument 1)", "run, load")
        table.add_row("🦋 Simulation Type", "(argument 2)", f"{IMPLEMENTED_TYPES}")
        table.add_row("🕹️  Interactive", "-i, --interactive", "Use flag to set True, otherwise False")
        table.add_row(long_line_divide, short_line_divide, long_line_divide)

        table.add_row("💯 Total number of timesteps","-t, --steps", f"Positive integer")
        table.add_row("⏳ Timestep duration (delta t)","-d, --deltat", f"Positive float")
        table.add_row(long_line_divide, short_line_divide, long_line_divide)

        table.add_row("🧮 Synchronous compute and render","-s, --sync", f"Use flag to set True, otherwise False")
        table.add_row("🧠 Cache history of timesteps","-c, --cache", f"True, False")
        table.add_row("📝 Write to NDJSON log","-l, --log", f"True, False")
        table.add_row("🎥 Save render to MP4 video","-v, --vid", f"Use flag to set True, otherwise False")
        table.add_row("🗄️  NDJSON log path", "--log_path\n OR --log_name\n OR --log_folder",f"Valid filepath string, ending in '.ndjson'")
        table.add_row("🎞️  MP4 video path", "--vid_path\n OR --vid_name\n OR --vid_folder", f"Valid filepath string, ending in '.mp4'")

        print("")
        self.console.print(table)
        print("")