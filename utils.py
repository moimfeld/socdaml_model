import argparse, os
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',       dest = 'train',    action = 'store_true',  default = False)
    parser.add_argument('--no-convert',  dest = 'convert',  action = 'store_false', default = True )
    parser.add_argument('--no-eval',     dest = 'eval',     action = 'store_false', default = True )
    parser.add_argument('--no-gen_data', dest = 'gen_data', action = 'store_false', default = True )
    parser.add_argument('--model',       dest = 'model',                            default = 'fc_model')
    return parser.parse_args()

def float_model_to_header(model, file_name = "c_headers/model.h"):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as file:
        # Generate top of file
        total_size = 0
        file.write('// List of model parameters:\n')
        file.write('//\n')
        for name, params in model.named_parameters():
            name = name.replace(".", "_")
            if 'weight' in name:
                if 'conv' in name:
                    params = params.squeeze(0).squeeze(0)
                file.write('// static const unsigned int ' + name + '_rows      ( = ' + str(params.shape[0]).rjust(4) + ' )\n')
                file.write('// static const unsigned int ' + name + '_columns   ( = ' + str(params.shape[1]).rjust(4) + ' )\n')
            elif 'bias' in name:
                file.write('// static const unsigned int ' + name + '_size        ( = ' + str(params.shape[0]).rjust(4) +' )\n')
            if 'weight' in name:
                size = (params.shape[0] * params.shape[1] * 4) / 1000
                file.write('// static const float {0}[{1}][{2}]'.format(name, params.shape[0], params.shape[1]).ljust(70) + '#       size = ' + '{0:.2f}'.format(size).rjust(8) + ' kB\n')
            elif 'bias' in name:
                size = (params.shape[0] * 4) / 1000
                file.write('// static const float {0}[{1}]'.format(name, params.shape[0]).ljust(70) + '#       size = ' + '{0:.2f}'.format(size).rjust(8) + ' kB\n')
            total_size += size
        file.write('//'.ljust(70) + '                ----------\n')
        file.write('//'.ljust(70) + '# Total Size = ' + '{0:.2f}'.format(total_size).rjust(8) + ' kB\n')
        file.write('\n')



        # Generate constants
        for name, params in model.named_parameters():
            name = name.replace(".", "_")
            if 'weight' in name:
                if 'conv' in name:
                    params = params.squeeze(0).squeeze(0)
                file.write('static const unsigned int ' + name + '_rows    = ' + str(params.shape[0]) + ';\n')
                file.write('static const unsigned int ' + name + '_columns = ' + str(params.shape[1]) + ';\n')
            elif 'bias' in name:
                file.write('static const unsigned int ' + name + '_size = ' + str(params.shape[0]) +';\n')

            # Write weights
            file.write('static const float ' + name)
            params = params.cpu().detach().numpy()
            if 'weight' in name:
                file.write('[' + str(params.shape[0]) + ']' + '[' + str(params.shape[1]) + '] = {')
                for i, row in enumerate(params):
                    file.write('{')
                    for j, param in enumerate(row):
                        if (j % 5) == 0:
                            file.write('\n')
                        file.write(str(param).rjust(15))
                        if j != len(row)-1:
                            file.write(', ')
                    if i != len(params)-1:
                        file.write('}, ')
                    else:
                        file.write('}\n')
                file.write('};\n')
            elif 'bias' in name:
                file.write('[' + str(params.shape[0]) + '] = {')
                for i, param in enumerate(params):
                    if (i % 5) == 0:
                        file.write('\n')
                    file.write(str(param).rjust(15))
                    if i != len(params)-1:
                        file.write(', ')
                file.write('};\n')

            file.write('\n')

def float_mnist_to_header(dataset, num_samples = 2, file_name = 'c_headers/images_labels.h'):
    images = []
    labels = []
    # get data from dataset
    for i, data in enumerate(dataset):
        if i == num_samples:
            break
        image, label = data
        image = image.squeeze(0).squeeze(0).flatten().cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        images.append(image)
        labels.append(label)

    # generate c-header file
    image_size = len(images[0])
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        # Write labels
        num_labels = len(labels)
        f.write("static const unsigned int num_labels = {0};\n".format(num_labels))
        f.write("static const unsigned char simplenet_labels[{0}] = ".format(num_labels))
        f.write("{")
        for j, number in enumerate(labels):
            f.write("{0}".format(number[0].astype(np.uint8)))
            if j < num_labels - 1:
                f.write(", ")
        if j < num_labels -1:
            f.write(",\n")
        f.write("};\n")


        # Write images
        f.write("static const unsigned int num_images = {0};\n".format(num_samples))
        f.write("static const unsigned int image_size = {0};\n".format(image_size))
        f.write("static const float simplenet_inputs[{0}][{1}] = ".format(num_samples, image_size))
        f.write("{")
        for i, image in enumerate(images):
            f.write("\t{")
            for j, number in enumerate(image):
                if j%5 == 0:
                    f.write("\n\t")
                f.write("{0}".format(number.astype(np.float)).rjust(20))
                if j < image_size - 1:
                    f.write(", ")
            f.write("\n\t}")
            if i < num_samples -1:
                f.write(",\n")
        f.write("\n};\n")