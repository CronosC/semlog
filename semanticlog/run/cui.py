# Import the necessary packages
from consolemenu import *
from consolemenu.items import *
from consolemenu.format import *

menu_format = MenuFormatBuilder().set_border_style_type(MenuBorderStyleType.HEAVY_BORDER) \
        .set_prompt("SELECT>") \
        .set_left_margin(4) \
        .set_right_margin(4)

        
menu = ConsoleMenu("Main", "Simple console ui for the main functionality of the semanticlog library.", formatter=menu_format)

echo = CommandItem("Run a console command",  "echo 'Hi'")

data_menu = ConsoleMenu("Data", "Data manipulation functionality.")


onto_menu = ConsoleMenu("Ontology", "Ontology manipulation functionality.")
learn_menu = ConsoleMenu("Learning", "Learning functionality.")

data_submenu = SubmenuItem("Data", data_menu, menu)
onto_submenu = SubmenuItem("Ontology", onto_menu, menu)
learn_submenu = SubmenuItem("Learning", learn_menu, menu)

# A FunctionItem runs a Python function when selected
#function_item = FunctionItem("Call a Python function", input, ["Enter an input"])

menu.append_item(data_submenu)
menu.append_item(onto_submenu)
menu.append_item(learn_submenu)

menu.show()