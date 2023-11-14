"""
Adapted from Source: https://github.com/columbia-robovision/html-visualization
Credits: Zhenjia Xu <xuzhenjia@cs.columbia.edu>

Script for generating html-table-visualizations for quick evaluations
"""


from pathlib import Path
import dominate
from typing import List
from pathlib import PurePath


def html_visualize(
    web_path, data, ids, cols, title="visualization", html_file_name: str = "index.html"
):
    """Visualization in html.

    Args:
        web_path: string; directory to save webpage. It will clear the old data!
        data: dict;
            key: {id}_{col}.
            value: Path / str / list of str
                - Path: Figure .png or .gif path
                - text: string or [string]
        ids: [string]; name of each row
        cols: [string]; name of each column
        title: (optional) string; title of the webpage (default 'visualization')
    """
    web_path = Path(web_path)
    web_path.parent.mkdir(parents=True, exist_ok=True)

    with dominate.document(title=title) as web:
        dominate.tags.h1(title)
        with dominate.tags.table(border=1, style="table-layout: fixed;"):
            with dominate.tags.tr():
                with dominate.tags.td(
                    style="word-wrap: break-word;",
                    halign="center",
                    align="center",
                    width="64px",
                ):
                    dominate.tags.p("id")
                for col in cols:
                    with dominate.tags.td(
                        style="word-wrap: break-word;",
                        halign="center",
                        align="center",
                    ):
                        dominate.tags.p(col)
            for idx in ids:
                with dominate.tags.tr():
                    bgcolor = "F1C073" if idx.startswith("train") else "C5F173"
                    with dominate.tags.td(
                        style="word-wrap: break-word;",
                        halign="center",
                        align="center",
                        bgcolor=bgcolor,
                    ):
                        for part in idx.split("_"):
                            dominate.tags.p(part)
                    for col in cols:
                        with dominate.tags.td(
                            style="word-wrap: break-word;", halign="center", align="top"
                        ):
                            key = f"{idx}_{col}"
                            if key in data:
                                value = data[key]
                            else:
                                value = ""
                            if isinstance(value, str) and (
                                value.endswith(".gif")
                                or value.endswith(".png")
                                or value.endswith(".jpg")
                            ):
                                dominate.tags.img(
                                    style="height:128px", src=data[f"{idx}_{col}"]
                                )
                            elif isinstance(value, list) and isinstance(value[0], str):
                                for val in value:
                                    dominate.tags.p(val)
                            else:
                                dominate.tags.p(str(value))

    with open(web_path / html_file_name, "w") as html_file:
        html_file.write(web.render())


def visualize_helper(
    tasks,
    dataset_path: Path,
    cols: List = None,
    html_file_name: str = "index.html",
    title: str = "dataset visualization",
):
    """
    Helper script for visualization
    """
    data = dict()
    ids = list()
    determine_cols = False
    if cols is None:
        cols = list()
        determine_cols = True
    for idx, dump_paths in enumerate(tasks):
        ids.append(str(idx))
        for col, path in dump_paths.items():
            if determine_cols and col not in cols:
                cols.append(col)
            if path is not None:
                if isinstance(path, PurePath):
                    data[f"{idx}_{col}"] = str(path.relative_to(dataset_path))
                else:
                    data[f"{idx}_{col}"] = path
    html_visualize(
        str(dataset_path),
        data,
        ids,
        cols,
        title=title,
        html_file_name=html_file_name,
    )
