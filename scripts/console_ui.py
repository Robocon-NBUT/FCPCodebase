from typing import Optional
from tabulate import tabulate
import numpy as np


class UI:
    @staticmethod
    def read_particle(prompt: str, str_options: list[str], dtype=str, interval=[-np.inf, np.inf]):
        if dtype is str and len(str_options) == 1:
            print(prompt, str_options[0], sep="")
            return 0, True
        elif dtype is int and interval[0] == interval[1]-1:
            print(prompt, interval[0], sep="")
            return interval[0], False

        while True:
            inp = input(prompt)
            if inp in str_options:
                return str_options.index(inp), True

            if dtype is not str:
                try:
                    inp = dtype(inp)
                    if interval[0] <= inp < interval[1]:
                        return inp, False
                except:
                    pass

            print("Error: illegal input! Options:", str_options,
                  f" or  {dtype}" if dtype != str else "")

    @staticmethod
    def input_num(prompt: str, min_num: int, max_num: int) -> int:
        while True:
            inp = input(prompt)
            try:
                num = int(inp)
                assert min_num <= num < max_num
                return num
            except:
                print(f"Error: 输入非法！输入的数字应该在 {min_num} 到 {max_num - 1} 之间")

    @staticmethod
    def print_table(
        data: list[list[str]],
        titles: Optional[list[str]] = None,
        colalign: Optional[list[str]] = None,
        numbering: Optional[list[bool]] = False,
        prompt: Optional[str] = None,
    ) -> Optional[tuple[int, int, int]]:
        """
        打印表格，支持列对齐方式和选项编号。

        Parameters
        ----------
        data : List[List[str]]
            表格数据，每一行是一个列表。
        titles : Optional[List[str]]
            列标题，默认为 None。
        colalign : Optional[List[str]]
            列对齐方式，默认为 None（左对齐）。支持 'left'（左对齐）、'right'（右对齐）、'center'（居中对齐）。
        numbering : bool
            是否为每行添加编号，默认为 False。
        prompt : Optional[str]
            提示信息，默认为 None。

        Returns
        -------
        Optional[Tuple[int, int, int]]
            返回用户选择的全局索引、列内索引和列号。如果未提供 prompt，则返回 None。
        """
        # 默认对齐方式为左对齐
        if colalign is None:
            colalign = ["left"] * len(data[0]) if data else []

        # 如果 numbering 为 True，为每行添加编号
        numbered_data = data
        count = len(data)
        if numbering:
            for i, row in enumerate(data):
                numbered_row = row.copy()
                if numbering[0]:
                    numbered_row[0] = f"{i}-{row[0]}"
                if numbering[1] and row[1]:
                    count += 1
                    numbered_row[1] = f"{i + len(data)}-{row[1]}"
                numbered_data[i] = numbered_row

        # 使用 tabulate 打印表格
        table = tabulate(numbered_data, headers=titles,
                         tablefmt="fancy_outline", colalign=colalign)
        print(table)

        # 如果未提供 prompt，直接返回
        if prompt is None:
            return None

        # 如果 numbering 为 True，处理用户输入
        if numbering:
            no_of_items = count
            index = UI.input_num(prompt, 0, no_of_items)

            # 返回全局索引、列内索引和列号
            return index, index % 15, index // 15
        else:
            print(prompt)
            return None

    @staticmethod
    def print_list(data: list[str], numbering=True, prompt: Optional[str] = None) -> Optional[tuple[int, str]]:
        if numbering:
            num_data = [[f"{i}-{item}"] for i, item in enumerate(data)]
        table = tabulate(num_data, tablefmt="fancy_outline")
        print(table)

        if prompt is None:
            return None

        idx = UI.input_num(prompt, 0, len(num_data))
        return idx, data[idx]
