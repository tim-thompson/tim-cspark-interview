# Time String Parser

This is a simple Python 3 parser for time strings. It takes a string and returns a datetime object.

The parser in this case is designed to parse splunk style strings in the format `now()[+/-][offset]@[snap_unit]` where `offset` is an integer with a time unit and `snap_unit` is one of `s`, `m`, `h`, `d`, `w`, `mon`, `y` which rounds down to the specified unit.

For example `now()-1h@d` would return a datetime object for the current time rounded down to the nearest day minus one hour.

The parse function is available in the `parse.py` file and can be imported and tested with the interactive python shell from this directory.

The tests use unittest and can be run by calling `python test_parse.py` from the command line with a recent version of Python 3 installed.