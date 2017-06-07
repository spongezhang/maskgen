from maskgen.plugins import loadPlugins, callPlugin, getOperation
from maskgen.image_wrap import  openImageFile
from maskgen.tool_set import validateAndConvertTypedValue
from maskgen import software_loader
import sys

def run_plugin(argv=None):
    import argparse
    import itertools
    if (argv is None):
        argv = sys.argv

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--plugin', help='name of plugin', required=True)
    parser.add_argument('--input', help='base image or video',required=True)
    parser.add_argument('--output', help='result image or video',  required=True)
    parser.add_argument('--arguments', nargs='+', default={},  help='Additional operation/plugin arguments e.g. rotation 60')
    args = parser.parse_args()

    op = software_loader.getOperation(getOperation(args.plugin)['name'])
    parsedArgs = dict(itertools.izip_longest(*[iter(args.arguments)] * 2, fillvalue=""))
    for key in parsedArgs:
        parsedArgs[key] = validateAndConvertTypedValue(key, parsedArgs[key], op,
                                                                skipFileValidation=False)

    loadPlugins()
    callPlugin(args.plugin, openImageFile(args.input), args.input, args.output, **parsedArgs)

if __name__ == "__main__":
    sys.exit(run_plugin())