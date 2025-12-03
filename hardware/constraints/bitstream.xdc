##-----------------------------------------------------------------------------
## File: bitstream.xdc
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Date          : 2025-01-23
## Description: Bitstream generation settings
##-----------------------------------------------------------------------------

## Bitstream compression (reduces file size)
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]

## Configuration rate (faster programming)
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]

## Enable USR_ACCESS (for design identification)
set_property BITSTREAM.CONFIG.USR_ACCESS TIMESTAMP [current_design]
