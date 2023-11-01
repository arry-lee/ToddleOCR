import ast
import importlib
import os
from collections import defaultdict

import torch
import yaml


def dynamic_import(module_path):
    module_name, _, class_name = module_path.rpartition(".")
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def get_all(module_file_path):
    with open(module_file_path, "r", encoding="utf-8") as file:
        module_ast = ast.parse(file.read())

    for node in module_ast.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    return [
                        elem.s for elem in node.value.elts if isinstance(elem, ast.Str)
                    ]

    all_names = []
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef):
            if node.name[0] != "_":
                all_names.append(node.name)

    return all_names


class Hub:
    def __init__(self, project_dir):
        cached_file = os.path.join(project_dir, "module_map.yml")
        if os.path.exists(cached_file):
            print(f"load {cached_file}")
            with open(cached_file, "r") as f:
                self.module_dict = yaml.safe_load(f)
        else:
            print("load module_map.py")
            # 自动根据配置文件寻找并精确导入对应类
            self.module_dict = defaultdict(list)

            # print(project_dir)
            # def get_all(module_file_path):
            #     with open(module_file_path, "r", encoding="utf-8") as file:
            #         module_ast = ast.parse(file.read())
            #
            #     all_names = set()
            #     for node in module_ast.body:
            #         if isinstance(node, ast.ClassDef):
            #             if node.name[0] != "_":
            #                 all_names.add(node.name)
            #
            #     return list(all_names)

            for root, dirs, files in os.walk(project_dir):
                for file in files:
                    if not file.startswith("_") and file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        name = file_path.removeprefix(os.path.dirname(project_dir))[
                            1:-3
                        ].replace(os.path.sep, ".")
                        try:
                            all = get_all(file_path)
                            # module = importlib.import_module(name=name)
                            # if hasattr(module, "__all__"):
                            #     all = getattr(module, "__all__")
                            # else:
                            #     print("[Warning] {} has no __all__".format(name))
                            #     all = get_all(file_path)
                        except Exception as e:
                            print("[Error] {}".format(file_path))
                            print(e)
                        else:
                            for cls in all:
                                self.module_dict[cls].append(name)
                                print(cls, name)
            with open(cached_file, "w", encoding="utf-8") as f:
                yaml.dump(dict(self.module_dict), f)

    def __getitem__(self, item):
        if "." in item:
            class_ = dynamic_import(item)
        else:
            module = importlib.import_module(self.module_dict[item][0])
            class_ = getattr(module, item)
        return class_

    def __call__(self, name, model_type=None):
        if "." in name:  # 详细导入
            class_ = dynamic_import(name)
        elif name in torch.nn.modules.__all__:  # torch.nn.modules 内置模型
            class_ = getattr(torch.nn.modules, name)
        else:  # 自定义模型
            if len(self.module_dict[name]) == 1:
                module = importlib.import_module(self.module_dict[name][0])
                class_ = getattr(module, name)
            else:
                if model_type is not None:
                    for one in self.module_dict[name]:
                        if one.rsplit(".", 1)[1].startswith(model_type.lower()):
                            module = importlib.import_module(one)
                            class_ = getattr(module, name)
                            break
                else:
                    raise ValueError("model_type is not specified")
        return class_
