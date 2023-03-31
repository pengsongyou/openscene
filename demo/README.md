# Real-time Interactive Demo
<p align="center">
  <img src="../media/demo.gif" width="80%" />
</p>

Here we present a **real-time**, **interactive**, **open-vocabulary** scene understanding tool. A user can type in an arbitrary query phrase like **`snoopy`** (rare object), **`somewhere soft`** (property), **`made of metal`** (material), **`where can I cook?`** (activity), **`festive`** (abstract concept) etc, and the correponding regions are highlighted.

## Prerequisites

### Environment
You can activate the `openscene` environment, or simply make sure the following package installed:
- `torch`
- `clip`
- `numpy`

The demo has been tested under Linux and Macbook.

## Run Demo
First, download the demo data:
```bash
cd demo
wget https://cvg-data.inf.ethz.ch/openscene/demo/demo_data.zip
unzip demo_data.zip
```

Second, set up the demo with following commands:
```bash
# compile gaps library
cd gaps
make 

# download and compile RNNets into gaps/pkgs/RNNets
cd pkgs
wget https://cvg-data.inf.ethz.ch/openscene/demo/RNNets.zip
unzip RNNets.zip
cd RNNets
make

# download and compile osview into gaps/apps/osview
# the executable will be in gaps/bin/x86_64/osview
cd ../apps
wget https://cvg-data.inf.ethz.ch/openscene/demo/osview.zip
unzip osview.zip
cd osview
make
```

Now, make sure you are under `demo/`, and you can simply run to have fun with the interactive demo:
```bash
./run_demo
```
You might need to edit the `run_demo` file to adapt the path to `osview`.
## Interactive Visualizer Usage
**Text query**: type words/sentences directly into the window, hit enter.

**Main commands**:
- `Left-click`: set the center point for camera zoom and rotate (important)
- `Right-click`: move the mesh with translation
- `Esc`: remove the current query
- `Alt-c`: change the color scheme
- `Ctrl-q`: quit

**Other commands**: Look in the Keyboard() function in gaps to see various alt- commands to toggle displays

## Customized Dataset
Coming soon.


## Troubleshooting
- If you get the error `OSError: Address already in use`, you might need to change another port in `clip_server.py`.
- For Mac users, you might need to change inside `run_demo` accordingly from `x86_64` to `arm64`.

For additional help, please refer to the code documentation or contact the author.