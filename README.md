# superpatcher

A patch for **Skylanders SuperChargers** (Xbox 360) that adds the content released only on the Wii U
version: two characters with their variants, their vehicles and the two related driver challenges.

The script contains no game files. It takes the content from the user's own extracted Wii U copy,
converts it to the Xbox 360 format and inserts it into the user's own Xbox 360 game files. Models,
textures, animations, audio, text and levels are adapted to the platform, so the figures are recognized
and playable, together with their vehicles and challenges.

## Compatibility

The patched game is meant for
[Skylanders: Core of Light](https://github.com/tsapZ/SkylandersCoreOfLight), the project that brings
the Xbox 360 versions of the series to PC. With the `--test-copy` option the script sets up a
separate, ready-to-use Core of Light installation: the original is left untouched and save files are
kept apart.

Tested with the European Xbox 360 release of SuperChargers (Italian and Spanish) on Core of Light
0.5.0-beta.9. The languages are detected from the Xbox 360 game, so the script also works with other
releases, which however have not been tested in game.

## Requirements

- Skylanders SuperChargers for Wii U, extracted: either the game folder (containing `content/archives`)
  or just the `archives` folder. The European release is needed to get every language.
- Skylanders SuperChargers for Xbox 360 installed with Core of Light, or the game's `archives` folder.
- Python 3 with `numpy` and `scipy`.
- `ffmpeg` with the `xma2` decoder, used to verify the converted audio.

The script does not decrypt or download anything and only reads the files it is given.

## Usage

```
python3 superpatcher.py --wiiu "<extracted SuperChargers Wii U>" \
    --x360 "<Core of Light folder with SuperChargers>" \
    --output <folder> [--test-copy "<new folder>"]
```

- `--output`: folder that receives the modified files, in `archives/`, and a `summary.txt`.
- `--test-copy`: creates a new installation next to the original one.
  - The original files are linked, not copied or modified.
  - In the `.toml` file it enables `portal_menu_nintendo`, the Core of Light option that shows these
    figures in the portal menu.
  - It adds a launcher `.bat` that uses separate save files.
  - It enlarges the hero and vehicle memory slots. Core of Light builds after 0.6.0-beta.3 read them
    from the `.toml` (`hero_data_slot_size` and `vehicle_slot_size` in `["Mods"]`), and the script
    writes them there. On older builds it patches the executable instead: the code is found by its
    surrounding bytes (checked on 0.5.0-beta.9 and 0.6.0-beta.2), and if it is not recognized the
    executable is left unchanged and the summary says so.
  - It raises the video memory budget for characters and vehicles (`VramBudget.igx`).
- `--languages es,it`: forces the languages instead of detecting them from the Xbox 360 game.
- `--patch-exe "<Core of Light folder>"`: only enlarges the memory slots of an existing installation.
  On builds that have the `.toml` settings it writes those and leaves the executable alone. On older
  builds it patches the executable, keeping the original as `superchargersrecompiled.exe.orig`: there
  a Core of Light update replaces the executable and the game then crashes when loading the added
  characters, so run it again after every update.

The conversion takes a few minutes. Every file written is checked again: format, archive indexes and,
for audio, a full decode with ffmpeg. If a check fails the script stops.

## How it works

- **Archives and resources**: `.pak` archives and `.igz` resources (Alchemy format) are converted from
  Wii U to Xbox 360.
  - Textures: from GX2 tiling to Xenos tiling.
  - Models: vertex and index buffers, vertex formats.
  - Text: only the languages present in the Xbox 360 game.
- **Figure registry**: the figures are added to the game's registry. Their driver challenges are made
  visible in the menu and linked to their tracks, with every resource they need.
- **Audio**: sound banks are re-encoded to XMA2, the Xbox 360 audio format, with an encoder written for
  this project (`xma.py`). Multichannel audio becomes multi-stream XMA. Each bank stays within the memory
  limit the game reserves for it.

## License

GNU Lesser General Public License 2.1 or later (see `LICENSE`). The audio encoder uses the coding tables
and frame structure of the WMA Pro decoder from [FFmpeg](https://ffmpeg.org/), distributed under the
same license.

## Legal notice

Hobby project, not affiliated with or endorsed by the owners of the games and trademarks mentioned. All
trademarks belong to their respective owners. Legitimate copies of both versions of the game are
required. The repository does not contain or distribute game files, keys or tools to decrypt them.
