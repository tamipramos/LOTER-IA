from dbcontroller import SQLiteController
import pyfiglet
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter



#########
## CLI ##
#########
# Autocomplete commands
commands = ["help", "train","update","populate", "predict", "exit"]
commands_completer = WordCompleter(commands, ignore_case=True)
commands2 = ["all", "cuponazo", "diario", "madre", "verano", "navidad", "padre", "sueldazo"]
commands_completer2 = WordCompleter(commands2, ignore_case=True)
# Name and title of the program
program_name = "Loter-IA"
ascii_art = pyfiglet.figlet_format(program_name)



########
## DB ##
########
# Initialize DB
db = SQLiteController("lottery.db")

# Create table
db.create_table("lottery", {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "type": "TEXT NOT NULL",
    "number": "TEXT",
    "series": "TEXT",
    "year": "INTEGER",
    "month": "INTEGER"
})
