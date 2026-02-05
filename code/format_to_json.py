import json
import warnings
from pathlib import Path
from collections import Counter
import zipfile

import pandas as pd

BASE_DIR = Path("/export/home/xwang/liu/RealHiTBench/data")

TABLES_DIR = BASE_DIR / "tables"
JSON_DIR = BASE_DIR / "json"

FORMAT_DIRS = {
    "tables": BASE_DIR / "tables",
    "csv": BASE_DIR / "csv",
    "markdown": BASE_DIR / "markdown",
    "latex": BASE_DIR / "latex",
    "html": BASE_DIR / "html",
    "image": BASE_DIR / "image",
    "json": BASE_DIR / "json",
}

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def is_valid_xlsx_file(path: Path) -> tuple[bool, str]:
    """
    检查文件是否是有效的xlsx文件
    返回: (是否有效, 错误信息)
    """
    # 检查扩展名
    if path.suffix.lower() != ".xlsx":
        return False, f"Not an xlsx file (extension: {path.suffix})"
    
    # 检查是否是ZIP文件（xlsx本质上是ZIP）
    try:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            files = zip_ref.namelist()
            # 检查是否包含必要的xlsx组件
            has_workbook = any('xl/workbook.xml' in f for f in files)
            has_worksheets = any('xl/worksheets/' in f for f in files)
            
            if not has_workbook:
                return False, "Missing xl/workbook.xml (corrupted xlsx structure)"
            if not has_worksheets:
                return False, "Missing xl/worksheets/ (corrupted xlsx structure)"
            
            return True, ""
    except zipfile.BadZipFile:
        return False, "Not a valid ZIP file"
    except Exception as e:
        return False, f"Error checking file: {str(e)}"


def stringify(x):
    if x is None:
        return None
    try:
        return str(x)
    except Exception:
        return repr(x)


def normalize_columns(cols):
    flat = []
    for c in cols:
        if isinstance(c, tuple):
            c = " | ".join([stringify(v) for v in c if v not in ("", None)])
        flat.append(stringify(c))

    counter = Counter()
    new_cols = []
    for c in flat:
        counter[c] += 1
        if counter[c] == 1:
            new_cols.append(c)
        else:
            new_cols.append(f"{c}_{counter[c]}")
    return new_cols


def safe_records(df):
    records = []
    for _, row in df.iterrows():
        item = {}
        for k, v in row.items():
            key = stringify(k)
            if pd.isna(v):
                item[key] = None
            else:
                item[key] = stringify(v)
        records.append(item)
    return records


def read_excel_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, dtype=object, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"Unreadable Excel workbook structure: {e}")


def xlsx_to_flat_json(xlsx_path: Path, json_path: Path):
    df = read_excel_robust(xlsx_path)
    df.columns = normalize_columns(df.columns)
    df = df.dropna(how="all")
    records = safe_records(df)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def get_file_stems(directory: Path, pattern: str = "*"):
    """获取目录下所有文件的stem（不含扩展名的文件名）"""
    if not directory.exists():
        return set()
    return {f.stem for f in directory.glob(pattern) if f.is_file()}


def analyze_file_discrepancies():
    """分析各文件夹之间的文件差异"""
    print("\n========== Analyzing File Discrepancies ==========")

    # 收集所有文件的stem
    file_stems = {}
    for name, path in FORMAT_DIRS.items():
        if path.exists():
            if name == "tables":
                file_stems[name] = get_file_stems(path, "*.xlsx")
            elif name == "csv":
                file_stems[name] = get_file_stems(path, "*.csv")
            elif name == "json":
                file_stems[name] = get_file_stems(path, "*.json")
            elif name == "image":
                file_stems[name] = get_file_stems(path, "*.png")
            else:
                file_stems[name] = get_file_stems(path)
        else:
            file_stems[name] = set()

    # 以tables作为基准
    base_stems = file_stems.get("tables", set())

    print(f"\nBase (tables): {len(base_stems)} files")

    # 检查每个格式与基准的差异
    for name, stems in file_stems.items():
        if name == "tables":
            continue

        missing = base_stems - stems  # tables有但该格式没有的
        extra = stems - base_stems  # 该格式有但tables没有的

        print(f"\n{name}:")
        print(f"  Total: {len(stems)}")
        print(f"  Missing from tables: {len(missing)}")
        if missing and len(missing) <= 10:
            for m in sorted(missing):
                print(f"    - {m}")
        elif missing:
            print(f"    (showing first 10)")
            for m in sorted(list(missing)[:10]):
                print(f"    - {m}")

        print(f"  Extra (not in tables): {len(extra)}")
        if extra and len(extra) <= 10:
            for e in sorted(extra):
                print(f"    + {e}")
        elif extra:
            print(f"    (showing first 10)")
            for e in sorted(list(extra)[:10]):
                print(f"    + {e}")


def count_files():
    """统计各文件夹的文件数量"""
    print("\n========== Dataset File Counts ==========")
    counts = {}
    for name, path in FORMAT_DIRS.items():
        if not path.exists():
            print(f"{name:10s}: 0 (missing)")
            counts[name] = 0
        else:
            files = list(path.glob("*"))
            file_count = len([f for f in files if f.is_file()])
            print(f"{name:10s}: {file_count}")
            counts[name] = file_count
    return counts


def check_all_files_in_tables():
    """检查tables目录下所有文件的有效性"""
    print("\n========== Checking Files in Tables Directory ==========")
    
    if not TABLES_DIR.exists():
        print("❌ Tables directory does not exist!")
        return [], [], []
    
    all_files = sorted(TABLES_DIR.glob("*"))
    all_files = [f for f in all_files if f.is_file()]
    
    valid_xlsx = []
    invalid_xlsx = []
    non_xlsx = []
    
    print(f"Total files in tables: {len(all_files)}")
    
    for file_path in all_files:
        # 首先检查扩展名
        if file_path.suffix.lower() != ".xlsx":
            non_xlsx.append((file_path.name, f"Not an xlsx file (extension: {file_path.suffix})"))
            continue
        
        # 检查是否是有效的xlsx
        is_valid, error_msg = is_valid_xlsx_file(file_path)
        
        if is_valid:
            valid_xlsx.append(file_path)
        else:
            invalid_xlsx.append((file_path.name, error_msg))
    
    # 输出统计
    print(f"\n✅ Valid xlsx files: {len(valid_xlsx)}")
    
    if non_xlsx:
        print(f"\n⚠️  Non-xlsx files: {len(non_xlsx)}")
        for name, reason in non_xlsx:
            print(f"    - {name}: {reason}")
    
    if invalid_xlsx:
        print(f"\n❌ Invalid/Corrupted xlsx files: {len(invalid_xlsx)}")
        for name, reason in invalid_xlsx:
            print(f"    - {name}: {reason}")
    
    return valid_xlsx, invalid_xlsx, non_xlsx


def main():
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    # 检查所有文件的有效性
    valid_xlsx, invalid_xlsx, non_xlsx = check_all_files_in_tables()
    
    total_valid = len(valid_xlsx)
    
    if total_valid == 0:
        print("\n❌ No valid xlsx files found to convert!")
        return

    success = 0
    failed = []

    print(f"\n========== Converting {total_valid} Valid XLSX to JSON ==========")
    for i, xlsx_path in enumerate(valid_xlsx, 1):
        json_path = JSON_DIR / f"{xlsx_path.stem}.json"
        try:
            xlsx_to_flat_json(xlsx_path, json_path)
            success += 1
            if i % 100 == 0:
                print(f"Processed {i}/{total_valid}...")
        except Exception as e:
            print(f"[ERROR] {xlsx_path.name}: {e}")
            failed.append((xlsx_path.name, str(e)))

    json_files = list(JSON_DIR.glob("*.json"))

    print("\n========== JSON Conversion Summary ==========")
    print(f"Valid xlsx files:    {total_valid}")
    print(f"JSON generated:      {len(json_files)}")
    print(f"Successful:          {success}")
    print(f"Failed:              {len(failed)}")
    
    if non_xlsx:
        print(f"Skipped (non-xlsx):  {len(non_xlsx)}")
    
    if invalid_xlsx:
        print(f"Skipped (corrupted): {len(invalid_xlsx)}")

    if failed:
        print("\n❌ Failed conversions:")
        for name, error in failed:
            print(f"   - {name}: {error}")
    
    if invalid_xlsx:
        print("\n⚠️  Corrupted xlsx files (skipped):")
        for name, error in invalid_xlsx:
            print(f"   - {name}: {error}")
    
    if non_xlsx:
        print("\n⚠️  Non-xlsx files (skipped):")
        for name, error in non_xlsx:
            print(f"   - {name}")

    # 统计文件数量
    counts = count_files()

    # 分析差异
    analyze_file_discrepancies()

    # 给出建议
    print("\n========== Recommendations ==========")
    
    if invalid_xlsx or non_xlsx:
        print("1. 清理 tables 目录:")
        if non_xlsx:
            print(f"   - {len(non_xlsx)} 个非xlsx文件应移除")
        if invalid_xlsx:
            print(f"   - {len(invalid_xlsx)} 个损坏的xlsx文件需要修复或移除")
    
    if failed:
        print("2. 处理转换失败的文件:")
        print("   - 检查这些文件是否可以从其他格式重建")

    max_count = max(counts.values())
    min_count = min(counts.values())
    if max_count != min_count:
        print("3. 文件数量不一致:")
        print("   - 这是正常的，因为跳过了无效文件")
        print("   - 确保所有有效的xlsx都成功转换为json即可")


if __name__ == "__main__":
    main()