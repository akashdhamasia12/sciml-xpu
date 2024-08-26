
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# command.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Main entry of the program
"""

from sciml_bench.core.statics import ABOUT_MSG
import click
import os
from pathlib import Path
import textwrap

from click.core import Context
from click.formatting import HelpFormatter
import sciml_bench.core.benchmark as Benchmark
import sciml_bench.core.dataset as Dataset
from sciml_bench.core.config import ProgramEnv
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
from sciml_bench.core.system import all_sys_info, format_info
from sciml_bench import __version__ as VERSION
from sciml_bench.core.utils import display_logo
from sciml_bench.core.utils import print_items
from sciml_bench.core.utils import extract_html_comments
from sciml_bench.core.utils import set_seeds

# init a global ProgramEnv instance
ENV = ProgramEnv(Path(__file__).parents[1] / 'etc/configs/config.yml')


class NaturalOrderGroup(click.Group):
    """ Force click to keep the order of commands """
    def list_commands(self, ctx):
        return self.commands.keys()
        
    def format_help(self, ctx: Context, formatter: HelpFormatter) -> None:
        display_logo()
        super().format_help(ctx, formatter)

@click.group(cls=NaturalOrderGroup, invoke_without_command=True)
@click.option('--version', is_flag=True, default=False, 
               help='\b\nDisplay the version of the software.')
def cli(version):
    if version:
        display_logo()
        click.echo(f' This is SciML-Bench Version. {VERSION}\n')

#####################################################################
# Primary Help                                                      #
#####################################################################


###################
# List Command 
###################

@cli.command('list', help='List datasets, and benchmarks.')
@click.argument('scope', default='all',
                type=click.Choice(['all', 'datasets', 'benchmarks'])) 
@click.option('--verify', is_flag=True, default=False,
              help='\b\nVerify existence of datasets, and modules of benchmarks.\n'\
                  'Default: False.')
@click.option('--deps', is_flag=True, default=False,
              help='\b\List benchmarks and their dependencies.\n'\
                  'Default: False.')                
def cmd_list(scope, verify, deps):
    """ sciml_bench list """

    display_logo()

    ####### The deps flag is only applicable to Benchmarks ######
    if deps == True:
        benchmark_names = ENV.list_benchmarks()
        for bench in benchmark_names:
            dataset_deps    = ENV.get_bench_datasets(bench)
            soft_deps       = ENV.get_bench_dependencies(bench)
            runnable_status = Benchmark.get_status(bench, ENV)
            dataset_deps    = ','.join(dataset_deps)
            soft_deps       = ','.join(soft_deps)
            bench_group     = ENV.get_bench_group(bench)
            print(f'  Benchmark: {bench}')
            print(f'\tSoftware Dependencies: {soft_deps}')
            print(f'\tDataset Dependencies : {dataset_deps}')
            print(f'\tRunnable Status      : {runnable_status}')
            print(f'\tBenchmark Group      : {bench_group}')
            print()
        return 

    # list datasets 
    if scope == 'all' or scope == 'datasets':
        dataset_names = ENV.list_datasets()
        dataset_statuses = None
        if verify:
            dataset_statuses = Dataset.get_status(dataset_names, ENV)
        print_items('Datasets', dataset_names, dataset_statuses)
    
    # list benchmarks
    if scope == 'all' or scope == 'benchmarks':
        benchmark_names = ENV.list_benchmarks()
        benchmark_status = None
        if verify:
            benchmark_status = Benchmark.get_status(benchmark_names, ENV)
        print_items('Benchmarks', benchmark_names, benchmark_status)
                
    print('\n')


###################
# Info Command 
###################

@cli.command(help='Obtain a detailed information about an entity.\n'\
                  'See the list command for available entities.')
@click.argument('entity')
def info(entity):
    """ sciml_bench info
    """
    # Decide whether the given entity is a benchmark or dataset:
    if entity in ENV.datasets.keys() or entity in ENV.benchmarks.keys():
      if entity in ENV.datasets.keys():
        info_path = Path(__file__).parents[1] / 'docs/markdown/datasets/' 
      else:
        info_path = Path(__file__).parents[1] / 'docs/markdown/benchmarks/' 
      content = extract_html_comments(str(info_path) + os.sep +  entity + '.md')
      if content:
        display_logo()
        lines = content.split('\n')
        wrapper = textwrap.TextWrapper(width=60, break_long_words=False, replace_whitespace=False)
        for line in lines:
            if len(line) > 60:
                line = '\n '.join(wrapper.wrap(line))
            print(f' {line}')
            
        return

    print(f' No information can be found on the entity {entity}.')
    print(f' Possible options are:\n')
    print_items('Benchmarks', ENV.list_benchmarks())
    print_items('Datasets', ENV.list_datasets())
    print()

###################
# Install Command 
###################


@cli.command(help='Install benchmark dependencies.')
@click.argument('benchmark_list')
def install(benchmark_list):
    """ sciml_bench list """
    Benchmark.install_benchmark_dependencies(benchmark_list, ENV)


###################
# Download Command 
###################

@cli.command(help='Download a dataset.')
@click.option('--dataset_dir', default=ENV.dataset_dir,
              help='\b\nRoot directory of datasets.\n'
                   'Default: dataset_dir in config.yml.\n')
@click.option('--mode', default='foreground',
                type=click.Choice(['foreground', 'background']),
                help='\b\nSets the downloading to foreground or background mode.\n'
                   'Default: foreground\n'
                )
@click.argument('dataset_names', nargs=-1)
def download(dataset_names, dataset_dir, mode):
    """ sciml_bench download """

    display_logo()

    # Pre-check if all datasets are registered
    for dataset_name in dataset_names:
        if not Dataset.is_dataset(dataset_name, ENV):
            return

    for dataset_name in dataset_names:
        print(f'Downloading the dataset {dataset_name}\n'\
                f'in {mode} mode.\n') 

        dataset_dir = Dataset.download(dataset_name, Path(dataset_dir), ENV, mode)
        if dataset_dir is None:
            print('Download Failed.')
            return  
        if mode == 'background':
            print(f'A log is available at\n'\
                f'    {ENV.output_dir}/download_logs/.\n')  
        else: 
            print(f'\nDownload complete.  Downloaded/synced the dataset to\n'\
                f'    {dataset_dir}. \n')
        



###################
# Run Command 
###################

@cli.command(help='Run a given benchmark on a training/inference mode.')
@click.option('--mode', required=False, default = 'training',
              type=click.Choice(['training', 'inference']),
              help='\b\nSets the mode to training or inference.\n'
                   'Default: training.')
@click.option('--model', required=False, multiple=False,
              help='\b\nSets the model(s) to be used (only for inference.)\n'
                   'If not specified, framework will attempt to find\n' 
                   'the model in the models directory\n'
                   'Default: None.')                       
@click.option('--dataset_dir', required=False, 
              help='\b\nDirectory for the dataset(s).\n'
                   'Default: dataset directory from the config file\n'
                             'if in training mode.\n')
@click.option('--output_dir', required=False,
              help='\b\nOutput directory for this run.\n'
                   'If not specified, outputs will be logged under\n' 
                   '        output_root_dir/benchmark_name/yyyymmdd/\n'
                   'where a yyyymmdd represents the current date\n'
                   'Use --output_dir=@foo to save outputs under\n'
                   '        output_root_dir/benchmark_name/foo/\n'
                   'If "@" is omitted, absolute path is assumed.\n')
@click.option('--monitor_on/--monitor_off', default=True,
              help='\b\nMonitor system usage during runtime.'
                   '\nDefault: Monitor On.')
@click.option('--monitor_interval', default=1.0, type=float,
              help='\b\nTime interval for system monitoring.'
                   '\nDefault: 1.0s.')
@click.option('--monitor_report_style', default='pretty',
              type=click.Choice(['pretty', 'yaml', 'hdf5']),
              help='\b\nReport style for system monitor.'
                   '\nDefault: Pretty.')
@click.option('--benchmark_specific', '-b', 'bench_args_list',
              type=(str, str), multiple=True,
              help='\b\nBenchmark-specific arguments.\n'
                   'Usage: -b key1 val1 -b key2 val2 ...')
@click.argument('benchmark_name')
def run(mode, model, dataset_dir, output_dir, monitor_on, 
        monitor_interval, monitor_report_style,
        bench_args_list, benchmark_name):
    """ sciml_bench run """
    

    # runtime input
    params_in = RuntimeIn(ENV, mode, model, benchmark_name, 
                          dataset_dir, output_dir, bench_args_list)


    if params_in.valid == False:
      print(params_in.error_msg)
      print()
      return

    # runtime output
    params_out = RuntimeOut(params_in.output_dir,
                          monitor_on=monitor_on,
                          monitor_interval=monitor_interval,
                          monitor_report_style=monitor_report_style)

    # create instance and run
    bench_types = ENV.get_bench_types(benchmark_name)
    bench_group = ENV.get_bench_group(benchmark_name)
    bench_run = None

    if mode in bench_types:
        if mode ==  'inference':
            bench_run = Benchmark.create_inference_instance(benchmark_name, bench_group)
        else:
            bench_run = Benchmark.create_training_instance(benchmark_name, bench_group)
    
    if (bench_run is None) or (mode not in bench_types):
        print(f'The benchmark {benchmark_name} does not support {mode}. Have you installed yet? Check with "sciml-bench list benchmarks --verify"')
        print(f'Terminating the execution')
        return 

    # Now try and launch
    try:    
        set_seeds()
        bench_run(params_in, params_out)
    except Exception as e:
        # kill system monitor thread
        params_out.system.abort()
        raise e
    
    # report monitor
    params_out.report()


###################
# Sysinfo Command 
###################


@cli.command(help='Display system information.')
def sysinfo():
    """ sciml_bench sysinfo """
    print(format_info(all_sys_info()))

###################
# About Command 
###################

@cli.command(help='About SciML-Bench.')
def about():
    """ sciml_bench about """
    about_info = ABOUT_MSG
    display_logo()
    print(about_info)


if __name__ == '__main__':
    cli(auto_envvar_prefix='SCIML_BENCH')

