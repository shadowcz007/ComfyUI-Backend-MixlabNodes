from importlib.metadata import PackageNotFoundError, version
import os,re

def find_requirements_files(directory):
    requirements_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'requirements.txt':
                requirements_files.append(os.path.join(root, file))
    return requirements_files


def split_version(line):
    pattern = r'([\w-]+)([<>=]+)([\d.]+)'
    matches = re.findall(pattern, line)
    if matches:
        package, operator, version = matches[0]
        return package, operator, version
    else:
        return line.strip(),None,'-'
    

# 指定目录
directory = './'

# 查找requirements.txt文件
requirements_files = find_requirements_files(directory)

# 统计依赖包
target_package={}


for file in requirements_files:

    dirname=os.path.dirname(file)
    basename=os.path.basename(dirname)
    with open(file) as f:
        for line in f:
            package, _operator, package_version=split_version(line)

            status=""

            if package:
                # print('#packag',package)
                if not package in target_package:
                    target_package[package]={}

                target_package[package][package_version]=basename

for package,vs in target_package.items():
    status=''
    basename=",".join(vs.values())
    try:   
        has_package_version = version(package)
        if has_package_version in vs:
            status='OK'
        else:
            if ",".join(vs.keys())=='-':
                status='OK'
            else:
                status=f'requirements:{",".join(vs.keys())}_installed:{has_package_version}'
            
    except PackageNotFoundError:
        status='Not found'

    if status!='OK':
        print('\033[91m')
        print(basename,'#',package,status)
        print("\033[0m")