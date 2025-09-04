---
hide:
  - toc
---


# Software Architecture

This codebase utilises extenal codebases for downloading source data, pre-processing and forecast delivery. The high-level software architecture diagram is as follows:

```puml
@startuml C4_Elements
!theme crt-amber
!include <bootstrap/bootstrap>
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Container.puml

left to right direction

title CANARI-ML Software Architecture

System_Ext(downloadAlias, "\n<$bi-cloud-arrow-down{scale=3}>\n\ndownload-toolbox", "Download data from various sources")
System_Ext(preprocessAlias, "\n<$bi-database{scale=3}>\n\npreprocess-toolbox", "Preprocess downloaded data, ready for training")
System(canariMLAlias, "\n<$bi-lightning{scale=3}>\n\ncanari-ml", "Main ML codebase, including netCDF generation, visualisation")
System_Ext(stacAlias, "\n<$bi-view-list{scale=3}>\n\nenvironmental-stac", "Process netCDF outputs for delivery of forecasts")

Rel(downloadAlias, preprocessAlias, "JSON Configuration")
Rel(preprocessAlias, canariMLAlias, "JSON Configuration")
Rel(canariMLAlias, stacAlias, "Convert netCDF to COGs & display on dashboard")
@enduml
```
