---
hide:
  - toc
---

# Software Architecture

This codebase utilises external codebases for downloading source data, pre-processing and forecast delivery. The high-level software architecture diagram is as follows:

```puml
@startuml
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Context.puml
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Container.puml
!include <bootstrap/bootstrap>
!include <material2.1.19/common>
!include <material2.1.19/monitor>
skinparam backgroundcolor transparent
left to right direction

title CANARI-ML Software Architecture

' Group toolboxes under env forecast
Boundary(efGroup, "Environmental Forecasting") {
    System_Ext(downloadAlias, "\n<$bi-cloud-arrow-down{scale=3}>\n\ndownload-toolbox", "Download data from various sources")
    System_Ext(preprocessAlias, "\n<$bi-database{scale=3}>\n\npreprocess-toolbox", "Preprocess downloaded data, reproject, normalise and prepare data for modelling")
}

' Group canari-ml and canari-ml-experiments
Boundary(canariGroup, "ML Codebase") {
    System(canariMLAlias, "\n<$bi-lightning{scale=3}>\n\ncanari-ml", "Main ML codebase, including netCDF generation, visualisation")
    System(canariMLExperimentsAlias, "\n<$bi-terminal{scale=2}> <$bi-filetype-yml{scale=2}>\n\ncanari-ml-experiments", "Experiment yaml configurations")
}

' Created separate group, else, diag was too complex for my liking
Boundary(efGroup2, "Environmental Forecasting") {
    System_Ext(stacAlias, "\n<$ma_monitor{scale=1.5}>\n\nenvironmental-stac", "Process netCDF outputs for delivery of forecasts")
}

Rel(downloadAlias, preprocessAlias, "JSON Configuration")
Rel(preprocessAlias, canariMLAlias, "JSON Configuration")
Rel(canariMLAlias, stacAlias, "Convert netCDF to COGs & display on dashboard")
Rel_R(canariMLExperimentsAlias, canariMLAlias, "Drives experiments")

url of downloadAlias is [[https://github.com/environmental-forecasting/download-toolbox]]
url of preprocessAlias is [[https://github.com/environmental-forecasting/preprocess-toolbox]]
url of canariMLAlias is [[https://github.com/canari-ml/canari-ml]]
url of canariMLExperimentsAlias is [[https://github.com/canari-ml/canari-ml-experiments]]
url of stacAlias is [[https://github.com/environmental-forecasting/environmental-stac-orchestrator]]

@enduml
```
