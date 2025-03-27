import ast
import hashlib
import inspect
import json
import os
import time
from datetime import datetime

from qtradex.common.utilities import it
from qtradex.core.ui_utilities import get_number, logo


def read_file(path):
    with open(path, "r") as handle:
        data = handle.read()
        handle.close()
    return data


def write_file(path, contents):
    with open(path, "w") as handle:
        handle.write(json.dumps(contents, indent=1))
        handle.close()


def get_path(bot):
    cache_dir = os.path.dirname(inspect.getfile(type(bot)))
    cache_dir = os.path.join(cache_dir, "tunes")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def generate_filename(bot):
    """
    generate filename unique to the bot's code
    plus the human-readable name of that code
    """
    # get the filename
    file = inspect.getfile(type(bot))
    source = read_file(file)
    # hash the ast of the code
    hashed = ast_to_hash(bot)
    # get the module name
    module_name = os.path.split(os.path.splitext(file)[0])[1]
    # generate a filename in the right place
    filename = os.path.join(get_path(bot), f"{module_name}_{hashed}.json")
    return filename, source


def save_tune(bot, identifier=None):
    filename, source = generate_filename(bot)

    # read the file
    try:
        contents = json.loads(read_file(filename))
    except FileNotFoundError:
        contents = {"source": source}
    if "source" not in contents:
        contents["source"] = source

    if identifier is None:
        identifier = time.ctime()
    else:
        if not isinstance(identifier, str):
            identifier = json.dumps(identifier)
        identifier += f"_{time.ctime()}"

    if bot.tune in contents.values():
        for k, v in contents.copy().items():
            # remove duplicate tunes only if they are unlabeled
            if v == bot.tune and "_" not in k:
                contents.pop(k)

    contents[identifier] = bot.tune

    write_file(filename, contents)


def from_iso_date(iso):
    return datetime.strptime(iso, "%a %b %d %H:%M:%S %Y").timestamp()


def load_tune(bot, key=None, sort="roi"):
    if isinstance(bot, str):
        path = get_path(bot)
        listdir = os.listdir(path)
        if bot not in listdir:
            raise KeyError(
                "Unknown bot id.  Try using `get_bots()` to find stored ids."
            )
        filename = os.path.join(path, bot)
    else:
        filename = generate_filename(bot)[0]
    try:
        contents = json.loads(read_file(filename))
    except FileNotFoundError:
        raise FileNotFoundError("The given bot has no saved tunes.")

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
        # get the latest key of this name
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
            raise KeyError(
                "Unknown tune key.  Try using `get_tunes(bot)` to find stored tunes."
            )

    return contents[key]["tune"]


def get_bots(bot):
    return sorted(os.listdir(bot if isinstance(bot, str) else get_path(bot)))


def get_tunes(bot):
    if isinstance(bot, str):
        path = get_path(bot)
        listdir = os.listdir(path)
        if bot not in listdir:
            raise KeyError(
                "Unknown bot id.  Try using `get_bots()` to find stored ids."
            )
        filename = os.path.join(path, bot)
    else:
        filename = generate_filename(bot)[0]
    try:
        contents = json.loads(read_file(filename))
    except FileNotFoundError:
        return []
    return contents


def ast_to_hash(instance):
    return len(instance.__class__().tune)


def choose_tune(bot, kind="any"):
    # allow bot to be both a filepath to a bot tune file or the bot itself
    if not isinstance(bot, str):
        bot = generate_filename(bot)[0]

    try:
        contents = json.loads(read_file(bot))
    except FileNotFoundError:
        raise FileNotFoundError("This bot has no saved tunes!")

    if kind == "tune":
        contents.pop("source")
    dispatch = {
        0: max(
            {k: v for k, v in contents.items() if k != "source"}.items(),
            key=lambda x: x[1]["results"]["roi"],
        )[0]
    }
    dispatch.update(enumerate(list(contents.keys()), start=1))

    logo()
    for k, v in dispatch.items():
        print(f"  {k}: {v}")
    option = dispatch[get_number(dispatch)]
    choice = contents[option]

    return choice["tune"] if kind == "tune" else choice


def main():
    logo(animate=True)
    path = os.path.join(os.getcwd(), "tunes")

    # sort by modified time
    algorithms = sorted(
        [os.path.join(path, i) for i in os.listdir(path)],
        key=os.path.getmtime,
        reverse=True,
    )

    if not algorithms:
        print("No saved tunes found!")
        return

    while True:
        logo()
        dispatch = dict(enumerate(algorithms + ["Exit"], start=1))
        print(it("yellow", "Bot save states, most recent first:"))
        for k, v in dispatch.items():
            print(f"  {k}: {os.path.splitext(os.path.split(v)[1])[0]}")
        choice = get_number(dispatch)

        if dispatch[choice] == "Exit":
            return

        tune = choose_tune(dispatch[choice])

        logo()
        if isinstance(tune, str):
            print(tune)
        else:
            print(json.dumps(tune, indent=4))
        input("\n\nEnter to continue.")
