import ast
import inspect
from pathlib import Path

py_files = sorted(Path("src/agent").rglob("*.py"))
for path in py_files:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    print(f"=== {path} ===")
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            init_node = next((n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__init__"), None)
            init_sig = ast.unparse(init_node.args) if init_node else "()"
            print(f"  class {node.name}({init_sig}) -> methods: {methods}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            print(f"  def {node.name}({ast.unparse(node.args)})")
