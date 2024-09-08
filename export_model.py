import argparse
from Model import model_configs
from Model import CompModel

prsr = argparse.ArgumentParser(
    description='''Export a trained model to C++ header files''')

# arguments for the training/test data locations and file names and config loading
prsr.add_argument('--model_config', '-mc',
                  help="Model config number"
                  , default=1)
prsr.add_argument('--ckpt_file', '-c',
                  help="Path to ckpt file")
prsr.add_argument('--output_dir', '-o',
                  help="Path to output the files")

args = prsr.parse_args()

if __name__ == "__main__":
    model_args = model_configs.main(int(args.model_config))

    # Create model from checkpoint
    model = CompModel.CompModel.load_from_checkpoint(args.ckpt_file, **model_args)

    # Export model to C++
    print("Exporting GainComputerParameters")
    model.static_comp.export(
        out_dir=args.output_dir,
        class_name='LA2AComputerModel',
        sub_class_name='LA2AComputerParameters',
    )

    print("Exporting GainSmootherParameters")
    model.gain_smooth.model.export(
        header_path='GainSmootherParameters.h',
        source_path='GainSmootherParameters.cpp',
        class_name='SmootherParameters',
    )

    print("Exporting MakeUpGainParameters")
    model.make_up.model.export(
        out_dir=args.output_dir,
        class_name='LA2AMakeUpModel',
        sub_class_name='LA2AMakeUpParameters',
    )
