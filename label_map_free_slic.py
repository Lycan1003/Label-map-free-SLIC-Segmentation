from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from skimage import color, img_as_ubyte, io


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class Cluster:
    """Cluster storing center position, Lab color, and local bit-mask."""

    _index = 1

    def __init__(self, h: int, w: int, L: float, A: float, B: float, S: int):
        self.no = Cluster._index
        Cluster._index += 1

        self.h = int(h)
        self.w = int(w)
        self.l = float(L)
        self.a = float(A)
        self.b = float(B)
        self.S = int(S)

        self.x = 0
        self.y = 0
        self.w_area = 0
        self.h_area = 0
        self.mask: list[int] = []

    def compute_area(self, img_h: int, img_w: int, scale: float = 1.7) -> None:
        """
        Faithful to the user's original implementation.
        Note: the width/height expansion below is intentionally preserved
        to reproduce the original results exactly.
        """
        size = max(1, math.ceil(scale * self.S))
        half_lo = (size - 1) // 2
        half_hi = size - 1 - half_lo
        x0, y0 = self.w - half_lo, self.h - half_lo
        x1, y1 = self.w + half_hi, self.h + half_hi
        self.x = max(0, x0)
        self.y = max(0, y0)
        x1 = min(img_w - 1, x1)
        y1 = min(img_h - 1, y1)

        self.w_area = max(1, x1 - self.x + 1)
        self.h_area = max(1, y1 - self.y + 1)

        if self.x + self.w_area < img_w:
            self.w_area = img_w - self.x
        if self.y + self.h_area < img_h:
            self.h_area = img_h - self.y

        self.w_area = min(self.w_area, img_w)
        self.h_area = min(self.h_area, img_h)

    def set_mask(self, cx: int, cy: int, val: bool) -> None:
        row = cy - self.y
        col = cx - self.x
        if 0 <= row < self.h_area and 0 <= col < self.w_area:
            self.mask[row * self.w_area + col] = 1 if val else 0

    def centroid(self) -> tuple[int, int] | None:
        cnt = sum(self.mask)
        if cnt == 0:
            return None
        sum_r = sum((idx // self.w_area) + self.y for idx, v in enumerate(self.mask) if v)
        sum_c = sum((idx % self.w_area) + self.x for idx, v in enumerate(self.mask) if v)
        return int(sum_r / cnt), int(sum_c / cnt)

    def update(self, nh: int, nw: int, L: float, A: float, B: float) -> None:
        self.h = int(nh)
        self.w = int(nw)
        self.l = float(L)
        self.a = float(A)
        self.b = float(B)


class SLICProcessor:
    def __init__(self, image_path: Path, k: int, m: float, search_scale: float = 1.7):
        self.image_path = Path(image_path)
        self.data = self.open_image(self.image_path)
        self.h, self.w = self.data.shape[:2]
        self.k = int(k)
        self.m = float(m)
        self.s = int(math.sqrt(self.h * self.w / self.k))
        self.search_scale = float(search_scale)
        self.clusters: list[Cluster] = []
        Cluster._index = 1

    @staticmethod
    def open_image(path: Path) -> np.ndarray:
        image = io.imread(path)
        return color.rgb2lab(image)

    @staticmethod
    def save_lab_image(path: Path, lab_array: np.ndarray) -> None:
        rgb_array = color.lab2rgb(lab_array)
        io.imsave(path, img_as_ubyte(rgb_array))

    def init_clusters(self) -> None:
        step = self.s
        for h in range(step // 2, self.h, step):
            for w in range(step // 2, self.w, step):
                L, A, B = self.data[h, w]
                self.clusters.append(Cluster(h, w, L, A, B, self.s))

    def get_distance(self, i: int, j: int, cluster: Cluster) -> float:
        L, A, B = self.data[i, j]
        dc = math.hypot(math.hypot(L - cluster.l, A - cluster.a), B - cluster.b)
        ds = math.hypot(i - cluster.h, j - cluster.w)
        return math.hypot(dc / self.m, ds / self.s)

    def get_gradient(self, i: int, j: int) -> float:
        i2 = min(i + 1, self.h - 2)
        j2 = min(j + 1, self.w - 2)
        gradient = self.data[i2, j2] - self.data[i, j]
        return float(np.sum(np.abs(gradient)))

    def move_clusters(self) -> None:
        for cluster in self.clusters:
            best_h, best_w = cluster.h, cluster.w
            best_g = self.get_gradient(cluster.h, cluster.w)
            for dh in (-1, 0, 1):
                for dw in (-1, 0, 1):
                    ni, nj = cluster.h + dh, cluster.w + dw
                    if 0 <= ni < self.h and 0 <= nj < self.w:
                        g = self.get_gradient(ni, nj)
                        if g < best_g:
                            best_g, best_h, best_w = g, ni, nj
            if (best_h, best_w) != (cluster.h, cluster.w):
                L, A, B = self.data[best_h, best_w]
                cluster.update(best_h, best_w, L, A, B)

    def assignment(self) -> None:
        grid_h = (self.h + self.s - 1) // self.s
        grid_w = (self.w + self.s - 1) // self.s
        cluster_grid = [[[] for _ in range(grid_w)] for _ in range(grid_h)]
        for cluster in self.clusters:
            grid_row = cluster.h // self.s
            grid_col = cluster.w // self.s
            cluster_grid[grid_row][grid_col].append(cluster)

        for cluster in self.clusters:
            cluster.compute_area(self.h, self.w, self.search_scale)
            cluster.mask = [0] * (cluster.h_area * cluster.w_area)

        for i in range(self.h):
            for j in range(self.w):
                pixel_grid_row = i // self.s
                pixel_grid_col = j // self.s
                best_cluster = None
                best_distance = math.inf

                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        r = pixel_grid_row + dr
                        c = pixel_grid_col + dc
                        if 0 <= r < grid_h and 0 <= c < grid_w:
                            for cluster in cluster_grid[r][c]:
                                row = i - cluster.y
                                col = j - cluster.x
                                if 0 <= row < cluster.h_area and 0 <= col < cluster.w_area:
                                    distance = self.get_distance(i, j, cluster)
                                    if distance < best_distance:
                                        best_distance = distance
                                        best_cluster = cluster

                if best_cluster is not None:
                    best_cluster.set_mask(j, i, True)

    def update_clusters(self) -> list[float]:
        distances = []
        for cluster in self.clusters:
            old_h, old_w = cluster.h, cluster.w
            centroid = cluster.centroid()
            if centroid:
                nh, nw = centroid
                dist_sq = (nh - old_h) ** 2 + (nw - old_w) ** 2
                distances.append(dist_sq)
                L, A, B = self.data[nh, nw]
                cluster.update(nh, nw, L, A, B)
        return distances

    def save_labels(self, output_csv: Path) -> None:
        label_array = np.zeros((self.h, self.w), dtype=int)
        for cluster in self.clusters:
            for idx, value in enumerate(cluster.mask):
                if value:
                    r, c = divmod(idx, cluster.w_area)
                    y = cluster.y + r
                    x = cluster.x + c
                    label_array[y, x] = cluster.no
        np.savetxt(output_csv, label_array, fmt="%d", delimiter=",")

    def save_boundary_image(self, output_image: Path) -> None:
        image_array = np.copy(self.data)
        for cluster in self.clusters:
            edge_h: dict[int, list[int]] = {}
            edge_w: dict[int, list[int]] = {}

            for idx, value in enumerate(cluster.mask):
                if value:
                    r, c = divmod(idx, cluster.w_area)
                    y = cluster.y + r
                    x = cluster.x + c
                    edge_h.setdefault(y, []).append(x)
                    edge_w.setdefault(x, []).append(y)

            for y, x_list in edge_h.items():
                min_x, max_x = min(x_list), max(x_list)
                image_array[y, min_x] = [0, 0, 0]
                image_array[y, max_x] = [0, 0, 0]

            for x, y_list in edge_w.items():
                min_y, max_y = min(y_list), max(y_list)
                image_array[min_y, x] = [0, 0, 0]
                image_array[max_y, x] = [0, 0, 0]

        self.save_lab_image(output_image, image_array)

    def run(self, max_iter: int = 10, ratio_threshold: float = 0.25) -> tuple[int, float | None]:
        """
        Faithful to the currently executed early-stop logic in the original file:
        - average squared displacement
        - diff-based stop: abs(curr - prev) < ratio_threshold * curr
        - no slope-based prediction
        """
        self.init_clusters()
        self.move_clusters()

        prev_avg_dist: list[float] = []
        stop_iter = max_iter
        final_avg_dist = None

        for i in range(max_iter):
            self.assignment()
            distances = self.update_clusters()

            if len(distances) == 0:
                stop_iter = i + 1
                final_avg_dist = 0.0
                break

            avg_dist = float(sum(distances) / len(distances))
            final_avg_dist = avg_dist

            prev_avg_dist.append(avg_dist)
            if len(prev_avg_dist) > 2:
                prev_avg_dist.pop(0)

            if len(prev_avg_dist) == 2:
                diff = abs(prev_avg_dist[1] - prev_avg_dist[0])
                ratio_stop = diff < ratio_threshold * avg_dist
            else:
                ratio_stop = False

            if ratio_stop:
                stop_iter = i + 1
                break

        return stop_iter, final_avg_dist


def process_image(
    image_path: Path,
    output_dir: Path,
    k: int,
    m: float,
    max_iter: int,
    ratio_threshold: float,
    search_scale: float,
) -> tuple[str, int, float | None]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    label_dir = output_dir / "labels"
    boundary_dir = output_dir / "boundaries"
    label_dir.mkdir(parents=True, exist_ok=True)
    boundary_dir.mkdir(parents=True, exist_ok=True)

    processor = SLICProcessor(image_path=image_path, k=k, m=m, search_scale=search_scale)
    stop_iter, final_avg_dist = processor.run(
        max_iter=max_iter,
        ratio_threshold=ratio_threshold,
    )

    # Preserve original naming style as much as possible: use basename with extension for logs,
    # but safe output filenames for images/csv use the stem.
    stem = image_path.stem
    processor.save_labels(label_dir / f"{stem}.csv")
    processor.save_boundary_image(boundary_dir / f"{stem}_boundary.png")

    return image_path.name, stop_iter, final_avg_dist


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path).resolve()


def collect_images(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label-map-free SLIC segmentation (repro-clean version).")
    parser.add_argument("--input-dir", type=str, default="data/images", help="Relative or absolute input image directory.")
    parser.add_argument("--output-dir", type=str, default="outputs_repro", help="Relative or absolute output directory.")
    parser.add_argument("--k", type=int, default=400, help="Number of superpixels.")
    parser.add_argument("--m", type=float, default=20.0, help="Compactness parameter.")
    parser.add_argument("--max-iter", type=int, default=10, help="Maximum number of iterations.")
    parser.add_argument("--ratio-threshold", type=float, default=0.25, help="Faithful to original executed code: diff < threshold * avg_dist.")
    parser.add_argument("--search-scale", type=float, default=1.7, help="Local square search scale relative to S.")
    parser.add_argument("--max-workers", type=int, default=8, help="Number of worker processes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    input_dir = resolve_path(args.input_dir, script_dir)
    output_dir = resolve_path(args.output_dir, script_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in: {input_dir}")

    log_rows = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_image,
                image_path,
                output_dir,
                args.k,
                args.m,
                args.max_iter,
                args.ratio_threshold,
                args.search_scale,
            )
            for image_path in image_paths
        ]

        for future in as_completed(futures):
            log_rows.append(future.result())

    log_rows.sort(key=lambda row: row[0])
    log_path = output_dir / "early_stop_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["image", "stop_iteration", "final_avg_dist_sq"])
        writer.writerows(log_rows)

    print(f"Processed {len(image_paths)} images.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
