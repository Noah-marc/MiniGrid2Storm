## Installing the project

The project uses uv for managing the dependencies. You can see the guidelines ofr installing on https://docs.astral.sh/uv/getting-started/installation/

Run `uv sync` in the project root for installing all necessary dependencies.

### Known installation issues

stormvogel uses the cairosvg library for visualization and uses it as a default dependency, even if the visualization tools are not used. At the time of writing, the imports are not defined lazily, so if you run into issues due to the cairo library make sure it is installed on your system. 

### Running the experiments

You can run the experiments with running `uv run experiments/1.0/[script_name].py` from the project root