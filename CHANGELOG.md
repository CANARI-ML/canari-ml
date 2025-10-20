## 0.0.1a2 (2025-10-20)

### Feat

- **preprocess**: Output compressed nc files in each step
- Use relative path symlink for final preprocess dir
- Add `hydra-submitit-launcher` dep for SLURM submission
- **wandb**: Updated tags, include model name
- **hydra**: Use overload postproc & add plot
- **nc**: Add postprocess config + nc output
- **preprocess**: Symlink to norm path w/ pred dset
- **torch**: Symlink to cache dir in train/pred
- **predict**: WIP HYDRA config+code for runs
- **hydra**: Add WandB logging support
- **hydra**: Tweaks to outputs w/ ensemble
- **hydra**: Use hydra conf for trainer as well
- **hydra**: Migrate to config driven training
- **hydra**: Add colour logging
- **hydra**: Code to skip previously run preproc steps
- **hydra**: Symlink preprocess outputs to main dir
- **hydra**: Add hash gen for each preprocess step
- **hydra**: Work towards non-default output dirs
- **hydra**: Proper logging, output dir definition
- **hydra**: Initial manual logging
- **hydra**: CLI entrypoints for download/preproc
- **hydra**: Add initial HYDRA config management

### Fix

- **postprocess**: Empty default './network_datasets' being created
- Add tensorboard dependency
- Move custom preprocess OmegaConf resolvers to utils
- **hydra**: Resolving interpolating by default print causing error
- slow argparse, temp fix
- **hydra**: Indentation of yaml only in logging
- **postprocess**: Setting type not overridden correctly
- **hydra**: Remove '.' from hydra search path in train
- **hydra**: Remove '.' from hydra search path in preprocess
- **hydra**: Adjust imports for new hydra/ entrypoint location
- Update pyproject to point to new GH org repo location
- **hydra**: Include pyproject update to new hydra location
- **hydra**: Incorrect download config location
- **hydra**: Config files not included in normal install
- **unet**: Forgot to commit for unet test
- Again, preprocess SLURM submission w/ multirun not working
- Preprocess SLURM submission w/ multirun not working
- Remove setting matplotlib backend
- **hydra**: preprocess --help cmd error
- **postprocess**: Incorrect single comment on postproc cli
- Re-add geospatial_bounds_crs to .nc outputs
- **dataloader**: Mem leak in cache zarr output
- **ncout**: Version info pointing to right place
- **model**: Incorrect type hint
- **hydra**: Register resolver before calling hydra func
- **pyproject**: Incorrect github dependency install format
- Changes to toolboxes & code for alternate output locations
- **pyproject**: Incorrect github dependency install format
- **zarr**: Limit zarr install version < 3
- **deps**: Pin numpy, min Python3.11 for hydra

### Refactor

- Remove icenet and numpy<2 deps
- **plot**: Add broadcast_forecast func from icenet
- **dataloader**: Strip out unused methods
- Working towards removing icenet deps
- **hydra**: Heavy refactor for individual cfg files
- **download**: Make consistent with other hydra init code
- **hydra**: Move cli/ subdir to hydra/
- **hydra**: Heavy changes, work from 1 exp file
- **hydra**: Rename 'main:' to 'input:'
- **hydra**: Move common configs to predict defaults dir
- **hydra**: train.run_name to train.name
- **hydra**: output layout, plotting, ensemble fixes
- **hydra**: Move hydra config print to utils
- **hydra**: Move train/predict configs to group
- **hydra**: Move train out into config group
- **ncout**: Move module path to postprocess
- **hydra**: Skip hash if name overridden
- **hydra**: Changes to make non-CLI preproc
- **download**: Update hydra config + no CLI
- **train**: Update ckpt output filename
- **hydra**: Streamline preprocess output locs
- **hydra**: Preprocess output locations & log
- **hydra**: Restructure config layout for hash
- **hydra**: Update output dir/path to dict
- **hydra**: Preprocess cmds in hydra config

## 0.0.1a1 (2025-06-19)

### Feat

- Add commitizen to manage commit + version
- **weight**: Add multi-level loss region weight

### Fix

- **deps**: pyproject dependencies
- Lag calc
- **dataloader**: Plotting with Torch dims order, not TF
- **unet**: Remove sigmoid at final layer
- **metrics**: Metric weighting in computation
- **loss**: Incorrect HuberLoss computation
- **loss**: Incorrect MSELoss computation
- **litmodule**: Incorrect MetricCollection cloning/copying

### Refactor

- **pyproject**: Update versioning
- Model target is now change in state
- **dataloader**: Make num_workers input arg
- **unet**: Cache padding in constructor for speed
- **loss**: Use cleaner factory approach
- **loss**: Tidy-up redundant code
- **loss**: Switch to MSELoss as default
- **unet**: Interpolate instead of ConvTranspose2d
- **litmodule**: Rename script and update import
- **litmodule**: Move common methods to Base class
- **litmodule**: Metric calculation in Lightning module
- **litmodule**: Update high level metric computation code
- **litmodule**: Improve metric set-up code, elegant?
- **litmodule**: Add batch_idx to step funcs

### Perf

- Add torch profiler stub, benchmark trainer flag

## 0.0.1a0 (2025-05-27)

### BREAKING CHANGE

- Some CLI flags are no longer accepted due to change.

### Feat

- Update pyproject for addition of unit testing
- **train**: Use adaptive AdamW learning rate
- **loader**: Use joblib for parallel output of Zarr cache
- **plot**: Show ticks in slider for visibility
- **plot**: Show forecast dates in ua700 error slider
- **plot**: Add slider and show in ua700 error if --plot-show
- **plot**: Enable outputting animation of ua700 error.
- **loader**: Output plots when writing Zarr cached datasets
- **preprocess**: Add ERA5 zg compute and preprocessing
- **unet**: Add dropouts to model
- Unused learning rate finder
- **loss**: Add Huber loss and switch to it
- **plot**: Visualise canari-ml prediction vs observation
- **output**: Create netCDF output from prediction
- **reproj**: Add omitted generalised reprojection
- **regrid**: Add CRS input to CRS reprojection
- **plot**: Plot raw numpy prediction output
- **regrid**: Add custom regrid from ERA5 to LAEA
- **predict**: Add prediction code
- **train**: Add training code
- **cli**: Add argument parsing for train/predict
- **model**: Add PytorchNetwork class
- **model**: Custom weight checkpointing class
- **model**: Add UNet model and lightning module
- **losses**: Add typical loss functions
- **metrics**: Add typically used metrics
- **dataloader**: Add torch dataloaders
- **dataloader**: Add code to generate samples
- **dataloader**: Initial Dask distributed data loader
- **dataloader**: Add CanariMLBaseDataLoader class
- **dataloader**: Update SerialLoader to write Zarr instead of tfrecords
- Cached tfrecord output based on icenet 'canari_ml_dataset_create'

### Fix

- **predict**: Dict selection if predicting test date
- **loss**: HuberLoss, and switch to using it
- **plot**: Bug introduced due to EPSG string based reprojection
- **reproject**: Raise exception if one process fails
- **unet**: Re-enable padding for varied dimension support
- Major fix - circular import
- **cli**: Correct duplicated help string in argparse
- **plot**: Increase resolution of ua700 error anim output
- **plot**: Correct ua700 comparison plot date indexing
- **cache**: Fix channel indexing including forecast init date
- Update get_implementation import
- **predict**: Update truth config file path
- **reproject**: Antimeridian gap in LAEA reprojection due to cubic resampling
- Loss being scaled to percentage when comparing
- **unet**: Fix error if shape not divisible by 16
- **regrid**: Coarsen to floating value shape
- **regrid**: Fix coarsen argument not being used

### Refactor

- **loader**: Pull date compute for input/pred
- **loader**: Use 0-1 cbar range for Zarr debug plots for consistency
- **plot**: Improve slider tick handling - more robust
- **loader**: Output xarray compat Zarr, tidy up.
- **reproject**: Tidy up reprojection functions
- **reproject**: Switch to Iris reprojection from rioxarray
- **reproject**: Switch to EPSG based src/target definitions
- **dataloader**: Remove unfinished DaskMultiWorkerLoader
- **predict**: minor unpacking variable for tidiness change
- **loader**: Switch to local CANARIMLBaseDataLoader instead of IceNetBaseDataLoader
- **plot**: Pull CLI parsing from icenet
- **loss**: Update dimension shifting, commented loss plotting code
- **unet**: Remove padding, switch to linear final layer instead of sigmoid
- Switch back to MSELoss, minor logging updates
- **plot**: Generalise plotting code
- Update Adam optimiser
- **plot**: Slight update, change linewidth
- **train**: Use Tensorboard instead of CSVLogger
- **reproj**: Rename regrid to reproject
- **unet**: Add sigmoid end, change upconv
- **dataloader**: Switch Zarr output to context manager

### Perf

- **train**: Switch to actual mixed precision for train
- **train**: Enable automatic mixed precision training
