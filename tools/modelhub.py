import ast
import importlib
import os
from collections import defaultdict

import yaml


def dynamic_import(module_path):
    module_name, _, class_name = module_path.rpartition(".")
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


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
            def get_all(module_file_path):
                with open(module_file_path, "r", encoding="utf-8") as file:
                    module_ast = ast.parse(file.read())

                all_names = set()
                for node in module_ast.body:
                    if isinstance(node, ast.ClassDef):
                        if node.name[0] != "_":
                            all_names.add(node.name)

                return list(all_names)

            for root, dirs, files in os.walk(project_dir):
                for file in files:
                    if not file.startswith("_") and file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        name = file_path.removeprefix(os.path.dirname(project_dir))[1:-3].replace(os.path.sep, ".")
                        try:
                            module = importlib.import_module(name=name)
                            if hasattr(module, "__all__"):
                                all = getattr(module, "__all__")
                            else:
                                all = get_all(file_path)
                        except Exception as e:
                            print(e)
                        for cls in all:
                            self.module_dict[cls].append(name)
            with open(cached_file, "w") as f:
                yaml.dump(dict(self.module_dict), f)

    def __getitem__(self, item):
        if "." in item:
            class_ = dynamic_import(item)
        else:
            module = importlib.import_module(self.module_dict[item][0])
            class_ = getattr(module, item)
        return class_

    def __call__(self, name, model_type=None):
        if "." in name:
            class_ = dynamic_import(name)
        else:
            if len(self.module_dict[name]) == 1:
                module = importlib.import_module(self.module_dict[name][0])
                class_ = getattr(module, name)
            else:
                if model_type is not None:
                    for one in self.module_dict[name]:
                        if one.startswith(model_type.lower()):
                            module = importlib.import_module(one)
                            class_ = getattr(module, name)
                            break
                else:
                    raise ValueError("model_type is not specified")
        return class_
