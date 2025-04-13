import ast
import hashlib
import inspect
import json
import os
import time
from datetime import datetime

from qtradex.common.utilities import it, read_file, write_file
from qtradex.core.ui_utilities import get_number, logo


def get_path(bot):
    """
    Get the directory path for storing tunes related to the bot.

    Parameters:
    - bot: The bot instance.

    Returns:
    - The path to the tunes directory.
    """
    cache_dir = os.path.dirname(inspect.getfile(type(bot)))
    cache_dir = os.path.join(cache_dir, "tunes")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def generate_filename(bot):
    """
    Generate a unique filename for the bot's tune based on its code and name.

    Parameters:
    - bot: The bot instance.

    Returns:
    - A tuple containing the generated filename and the source code of the bot.
    """
    file = inspect.getfile(type(bot))
    source = read_file(file)
    hashed = ast_to_hash(bot)  # Hash the bot's tune for uniqueness
    module_name = os.path.split(os.path.splitext(file)[0])[1]
    filename = os.path.join(get_path(bot), f"{module_name}_{hashed}.json")
    return filename, source


def save_tune(bot, identifier=None):
    """
    Save the current tune of the bot to a JSON file.

    Parameters:
    - bot: The bot instance.
    - identifier: Optional identifier for the tune.
    """
    filename, source = generate_filename(bot)

    # Attempt to read existing contents from the file
    try:
        contents = json.loads(read_file(filename))
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        contents = {"source": source}  # Initialize with source if file doesn't exist

    if "source" not in contents:
        contents["source"] = source

    # Generate a unique identifier if not provided
    if identifier is None:
        identifier = time.ctime()
    else:
        if not isinstance(identifier, str):
            identifier = json.dumps(identifier)
        identifier += f"_{time.ctime()}"

    # Remove duplicate tunes if they are unlabeled
    if bot.tune in contents.values():
        for k, v in contents.copy().items():
            if v == bot.tune and "_" not in k:
                contents.pop(k)

    contents[identifier] = bot.tune  # Save the new tune
    write_file(filename, contents)  # Write updated contents to file


def from_iso_date(iso):
    """
    Convert an ISO date string to a UNIX timestamp.

    Parameters:
    - iso: The ISO date string.

    Returns:
    - The corresponding UNIX timestamp.
    """
    return datetime.strptime(iso, "%a %b %d %H:%M:%S %Y").timestamp()


def load_tune(bot, key=None, sort="roi"):
    """
    Load a specific tune for the bot from the saved tunes.

    Parameters:
    - bot: The bot instance or its identifier.
    - key: Optional key to specify which tune to load.
    - sort: The sorting criteria for selecting the tune.

    Returns:
    - The loaded tune.
    """
    if isinstance(bot, str):
        path = get_path(bot)
        listdir = os.listdir(path)
        if bot not in listdir:
            raise KeyError("Unknown bot id. Try using `get_bots()` to find stored ids.")
        filename = os.path.join(path, bot)
    else:
        filename = generate_filename(bot)[0]

    try:
        contents = json.loads(read_file(filename))
    except FileNotFoundError:
        raise FileNotFoundError("The given bot has no saved tunes.")

    # Determine the key to load based on sorting criteria
    if key is None:
        if sort == "roi":
            key = max(
                {k: v for k, v in contents.items() if k != "source"}.items(),
                key=lambda x: x[1]["results"]["roi"],
            )[0]
        else:
            key = max(
            [
                i
                for i in contents.keys()
                if i != "source" and i.rsplit("_", 1)[0] == "BEST ROI TUNE"
            ],
            key=lambda x: from_iso_date(x.rsplit("_", 1)[1]),
        )

    if key not in contents:
        # Get the latest key of this name if the specified key is not found
        latest = max(
            [
                (
                    from_iso_date(i.rsplit("_", 1)[1]),
                    i.rsplit("_", 1)[1],
                )
                for i in contents.keys()
                if i != "source" and i.rsplit("_", 1)[0] == key
            ],
            key=lambda x: x[0],
        )
        key = [i for i in contents.keys() if i.endswith(latest[1])]
        if key:
            key = key[0]
        else:
            raise KeyError("Unknown tune key. Try using `get_tunes(bot)` to find stored tunes.")

    return contents[key]["tune"]  # Return the loaded tune


def get_bots(bot):
    """
    Get a sorted list of bot identifiers.

    Parameters:
    - bot: The bot instance or its identifier.

    Returns:
    - A sorted list of bot identifiers.
    """
    return sorted(os.listdir(bot if isinstance(bot, str) else get_path(bot)))


def get_tunes(bot):
    """
    Retrieve all tunes associated with a specific bot.

    Parameters:
    - bot: The bot instance or its identifier.

    Returns:
    - A list of tunes associated with the bot.
    """
    if isinstance(bot, str):
        path = get_path(bot)
        listdir = os.listdir(path)
        if bot not in listdir:
            raise KeyError("Unknown bot id. Try using `get_bots()` to find stored ids.")
        filename = os.path.join(path, bot)
    else:
        filename = generate_filename(bot)[0]

    try:
        contents = json.loads(read_file(filename))
    except FileNotFoundError:
        return []  # Return an empty list if no tunes are found

    return contents  # Return the contents of the tunes


def ast_to_hash(instance):
    """
    Generate a hash based on the bot's tune.

    Parameters:
    - instance: The bot instance.

    Returns:
    - A hash value representing the tune.
    """
    return len(instance.__class__().tune)  # Use the length of the tune as a simple hash


def choose_tune(bot, kind="any"):
    """
    Allow the user to choose a tune from the available options.

    Parameters:
    - bot: The bot instance or the path to a tune file.
    - kind: The type of choice to return (either "tune" or "any").

    Returns:
    - The chosen tune or choice based on the specified kind.
    """
    # Allow bot to be both a filepath to a bot tune file or the bot itself
    if not isinstance(bot, str):
        bot = generate_filename(bot)[0]

    try:
        contents = json.loads(read_file(bot))
    except FileNotFoundError:
        raise FileNotFoundError("This bot has no saved tunes!")

    if kind == "tune":
        contents.pop("source")  # Remove source if only tunes are needed

    # Create a dispatch dictionary for user selection
    dispatch = {
        0: max(
            {k: v for k, v in contents.items() if k != "source"}.items(),
            key=lambda x: x[1]["results"]["roi"],
        )[0]
    }
    dispatch.update(enumerate(list(contents.keys()), start=1))

    logo()  # Display logo
    for k, v in dispatch.items():
        print(f"  {k}: {v}")  # Print available options

    option = dispatch[get_number(dispatch)]  # Get user choice
    choice = contents[option]  # Retrieve the chosen tune

    return choice["tune"] if kind == "tune" else choice  # Return the appropriate choice


def main():
    """
    Main function to run the tune management interface.
    """
    logo(animate=True)
    path = os.path.join(os.getcwd(), "tunes")

    # Sort saved tunes by modified time
    algorithms = sorted(
        [os.path.join(path, i) for i in os.listdir(path)],
        key=os.path.getmtime,
        reverse=True,
    )

    if not algorithms:
        print("No saved tunes found!")
        return

    while True:
        logo()  # Display logo
        dispatch = dict(enumerate(algorithms + ["Exit"], start=1))
        print(it("yellow", "Bot save states, most recent first:"))
        for k, v in dispatch.items():
            print(f"  {k}: {os.path.splitext(os.path.split(v)[1])[0]}")  # Display saved tunes

        choice = get_number(dispatch)  # Get user choice

        if dispatch[choice] == "Exit":
            return  # Exit the loop if the user chooses to exit

        tune = choose_tune(dispatch[choice])  # Choose a tune based on user selection

        logo()  # Display logo again
        if isinstance(tune, str):
            print(tune)  # Print the tune if it's a string
        else:
            print(json.dumps(tune, indent=4))  # Print the tune in JSON format

        input("\n\nPress Enter to continue.")  # Wait for user input before continuing


if __name__ == "__main__":
    main()  # Run the main function when the script is executed
