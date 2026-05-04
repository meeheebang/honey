"""
SEM 이미지에서 각 구조물의 내부(검은 점)와 외곽 원 둘레를 계산합니다.
중심점 기반으로 반지름을 찾기 때문에 전역 Hough보다 누락/오검출에 강합니다.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import csv


InnerContour = np.ndarray


def center_dark_contrast_ok(gray: np.ndarray, cx: float, cy: float) -> bool:
    """중심이 주변 고리보다 충분히 어두운지 검사."""
    h, w = gray.shape[:2]
    mask_center = np.zeros((h, w), np.uint8)
    mask_ring = np.zeros((h, w), np.uint8)
    c = (int(round(cx)), int(round(cy)))
    cv2.circle(mask_center, c, 3, 255, -1)
    cv2.circle(mask_ring, c, 12, 255, -1)
    cv2.circle(mask_ring, c, 7, 0, -1)
    center_vals = gray[mask_center > 0]
    ring_vals = gray[mask_ring > 0]
    if center_vals.size < 10 or ring_vals.size < 20:
        return False
    return float(np.mean(ring_vals) - np.mean(center_vals)) > 18.0


def estimate_nm_per_pixel_from_scale_bar(gray: np.ndarray, known_nm: float = 200.0) -> float | None:
    h, w = gray.shape[:2]
    strip = gray[int(h * 0.72) : h, 0 : int(w * 0.45)]
    if strip.size == 0:
        return None
    _, bw = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(bw)) > 127:
        bw = 255 - bw
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 11), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_len = 0
    for c in contours:
        _, _, cw, ch = cv2.boundingRect(c)
        if ch < 3 or cw < 30:
            continue
        if cw > ch * 2.5 and cw > best_len:
            best_len = cw
    if best_len < 20:
        return None
    return known_nm / float(best_len)


def detect_centers(gray: np.ndarray) -> list[tuple[float, float]]:
    """검은 중심점을 blob으로 검출하고 중복 제거."""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 500
    params.filterByCircularity = True
    params.minCircularity = 0.45
    params.filterByInertia = False
    params.filterByConvexity = False
    detector = cv2.SimpleBlobDetector_create(params)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    keypoints = detector.detect(blur)
    points = [(float(k.pt[0]), float(k.pt[1])) for k in keypoints]
    if not points:
        return []

    # 이미지 테두리/스케일 바/좌상단 n 라벨 주변 제외
    h, w = gray.shape[:2]
    filtered: list[tuple[float, float]] = []
    for x, y in points:
        if x < 14 or y < 14 or x > w - 22 or y > h - 14:
            continue
        if x < w * 0.16 and y < h * 0.22:
            continue
        if y > h * 0.82 and x < w * 0.45:
            continue
        if not center_dark_contrast_ok(gray, x, y):
            continue
        filtered.append((x, y))

    # 근접 중복 병합
    merged: list[tuple[float, float]] = []
    min_sep = 24.0
    for x, y in sorted(filtered, key=lambda p: (p[1], p[0])):
        keep = True
        for mx, my in merged:
            if math.hypot(x - mx, y - my) < min_sep:
                keep = False
                break
        if keep:
            merged.append((x, y))
    return merged


def radial_profiles(gray: np.ndarray, cx: float, cy: float, r_min: int, r_max: int) -> tuple[np.ndarray, np.ndarray]:
    """지정 중심에서 반지름별 평균 밝기/기울기 프로파일 계산."""
    h, w = gray.shape[:2]
    angles = np.linspace(0, 2 * np.pi, 90, endpoint=False, dtype=np.float32)
    radii = np.arange(r_min, r_max + 1, dtype=np.float32)
    vals = []
    for r in radii:
        xs = cx + r * np.cos(angles)
        ys = cy + r * np.sin(angles)
        x0 = np.clip(np.floor(xs).astype(np.int32), 0, w - 1)
        y0 = np.clip(np.floor(ys).astype(np.int32), 0, h - 1)
        vals.append(gray[y0, x0].mean())
    intensity = np.array(vals, dtype=np.float32)
    grad = np.gradient(intensity)
    return intensity, grad


def estimate_radii(gray: np.ndarray, cx: float, cy: float) -> tuple[float | None, float | None]:
    """내부/외곽 반지름 추정. 내부는 급상승 에지, 외곽은 그 이후 최대 강한 에지."""
    h, w = gray.shape[:2]
    max_possible = int(min(cx, cy, w - 1 - cx, h - 1 - cy))
    if max_possible < 8:
        return None, None

    r_min = 1
    r_max = min(max_possible, 45)
    intensity, grad = radial_profiles(gray, cx, cy, r_min, r_max)
    radii = np.arange(r_min, r_max + 1, dtype=np.int32)

    # 내부: 중심 검은 점 경계(밝기 상승)
    inner_search = (radii >= 2) & (radii <= 14)
    if not np.any(inner_search):
        return None, None
    g_inner = grad.copy()
    g_inner[~inner_search] = -1e9
    inner_idx = int(np.argmax(g_inner))
    if g_inner[inner_idx] < 1.2:
        return None, None
    r_inner = float(radii[inner_idx])

    # 외곽: 내부보다 바깥에서 나타나는 강한 상승 에지
    outer_start = int(max(r_inner + 6, 12))
    outer_end = min(r_max, 55)
    outer_search = (radii >= outer_start) & (radii <= outer_end)
    if not np.any(outer_search):
        return r_inner, None
    g_outer = grad.copy()
    g_outer[~outer_search] = -1e9
    outer_idx = int(np.argmax(g_outer))
    if g_outer[outer_idx] < 0.8:
        return r_inner, None
    r_outer = float(radii[outer_idx])
    if r_outer <= r_inner + 2:
        return r_inner, None
    return r_inner, r_outer


def ring_edge_score(grad_mag: np.ndarray, cx: float, cy: float, r: float) -> float:
    h, w = grad_mag.shape[:2]
    angles = np.linspace(0, 2 * np.pi, 120, endpoint=False, dtype=np.float32)
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)
    valid = (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
    if not np.any(valid):
        return -1.0
    x0 = np.clip(np.floor(xs[valid]).astype(np.int32), 0, w - 1)
    y0 = np.clip(np.floor(ys[valid]).astype(np.int32), 0, h - 1)
    vals = grad_mag[y0, x0]
    if vals.size < 30:
        return -1.0
    return float(np.mean(vals))


def refine_outer_center(gray: np.ndarray, cx: float, cy: float, r_outer: float) -> tuple[float, float]:
    """고정 반지름에서 원주 에지 강도가 최대가 되는 중심으로 외곽 중심 보정."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    best_cx, best_cy = cx, cy
    best_score = ring_edge_score(grad_mag, cx, cy, r_outer)
    for dx in np.linspace(-3.0, 3.0, 13):
        for dy in np.linspace(-3.0, 3.0, 13):
            tx = cx + float(dx)
            ty = cy + float(dy)
            score = ring_edge_score(grad_mag, tx, ty, r_outer)
            if score > best_score:
                best_score = score
                best_cx, best_cy = tx, ty
    return best_cx, best_cy


def trace_inner_contour(gray: np.ndarray, cx: float, cy: float, r_inner_hint: float) -> tuple[InnerContour | None, float | None]:
    """중심 주변에서 실제 검은 영역 contour를 추적하고 둘레를 반환."""
    h, w = gray.shape[:2]
    pad = int(max(10, min(20, r_inner_hint * 3.5)))
    x0 = max(0, int(cx) - pad)
    y0 = max(0, int(cy) - pad)
    x1 = min(w, int(cx) + pad + 1)
    y1 = min(h, int(cy) + pad + 1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None, None

    # 로컬 임계값으로 검은 중심 영역 분리
    t = int(np.percentile(roi, 22))
    _, bw = cv2.threshold(roi, t, 255, cv2.THRESH_BINARY_INV)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None

    local_cx = float(cx - x0)
    local_cy = float(cy - y0)
    best: np.ndarray | None = None
    best_area = -1.0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 8.0 or area > 350.0:
            continue
        inside = cv2.pointPolygonTest(c, (local_cx, local_cy), False)
        if inside < 0:
            continue
        if area > best_area:
            best_area = area
            best = c
    if best is None:
        return None, None

    perim = float(cv2.arcLength(best, True))
    best_global = best.copy()
    best_global[:, 0, 0] += x0
    best_global[:, 0, 1] += y0
    return best_global, perim


def draw_overlay(
    bgr: np.ndarray,
    results: list[
        tuple[
            float,
            float,
            float | None,
            float | None,
            float | None,
            InnerContour | None,
            float | None,
            float | None,
        ]
    ],
    nm_per_px: float | None,
    line_thickness: int,
) -> np.ndarray:
    out = bgr.copy()
    for cx, cy, r_inner, r_outer, inner_perim_px, inner_contour, ocx, ocy in results:
        c = (int(round(cx)), int(round(cy)))  # inner center
        if inner_contour is not None:
            cv2.polylines(out, [inner_contour], True, (0, 255, 0), line_thickness, lineType=cv2.LINE_AA)
        elif r_inner is not None:
            cv2.circle(out, c, int(round(r_inner)), (0, 255, 0), line_thickness, lineType=cv2.LINE_AA)

        if inner_perim_px is not None:
            txt = f"in C={inner_perim_px * nm_per_px:.1f} nm" if nm_per_px else f"in C={inner_perim_px:.1f} px"
            cv2.putText(out, txt, (c[0] + 4, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 220, 0), 1, cv2.LINE_AA)
        if r_outer is not None:
            center_outer = (int(round(ocx if ocx is not None else cx)), int(round(ocy if ocy is not None else cy)))
            cv2.circle(out, center_outer, int(round(r_outer)), (0, 0, 255), line_thickness, lineType=cv2.LINE_AA)
            c_outer = 2.0 * math.pi * r_outer
            txt = f"out C={c_outer * nm_per_px:.1f} nm" if nm_per_px else f"out C={c_outer:.1f} px"
            cv2.putText(
                out,
                txt,
                (center_outer[0] + 4, center_outer[1] + int(round(r_outer)) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            dx = (ocx - cx) if ocx is not None else 0.0
            dy = (ocy - cy) if ocy is not None else 0.0
            txt_shift = f"shift dx={dx:.2f}, dy={dy:.2f}px"
            cv2.putText(out, txt_shift, (c[0] + 4, c[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 220, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SEM 원형 구조 내/외곽 둘레 측정")
    parser.add_argument("image", nargs="?", type=Path, default=Path(__file__).parent / "sem_input.png")
    parser.add_argument("-o", "--output", type=Path, default=Path(__file__).parent / "sem_measured.png")
    parser.add_argument("--no-scale", action="store_true", help="스케일 바 자동 추정 비활성화")
    parser.add_argument("--nm-per-pixel", type=float, default=None, help="직접 보정: 1픽셀당 nm")
    parser.add_argument("--line-thickness", type=int, default=1, help="원 둘레 선 두께 (기본 1)")
    args = parser.parse_args()

    gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"이미지를 열 수 없습니다: {args.image}")

    centers = detect_centers(gray)
    if not centers:
        raise SystemExit("중심 검은 점을 찾지 못했습니다. 이미지 대비를 확인하세요.")

    results: list[
        tuple[
            float,
            float,
            float | None,
            float | None,
            float | None,
            InnerContour | None,
            float | None,
            float | None,
        ]
    ] = []
    for cx, cy in centers:
        r_in, r_out = estimate_radii(gray, cx, cy)
        # 내부 홀 반지름이 비정상적으로 큰/작은 잡점은 버림
        if r_in is None or r_in < 3.0 or r_in > 7.5:
            continue
        inner_contour, inner_perim_px = trace_inner_contour(gray, cx, cy, r_in)
        if inner_perim_px is None:
            inner_perim_px = 2.0 * math.pi * r_in
        # 외곽은 데이터 특성상 대체로 22~32 px 범위
        if r_out is not None and not (22.0 <= r_out <= 32.0):
            r_out = None
        if r_out is not None:
            ocx, ocy = refine_outer_center(gray, cx, cy, r_out)
        else:
            ocx, ocy = None, None
        results.append((cx, cy, r_in, r_out, inner_perim_px, inner_contour, ocx, ocy))

    # 무조건 채우는 fallback은 오인식을 만들 수 있으므로 제거

    nm_per_px: float | None = args.nm_per_pixel
    if nm_per_px is None and not args.no_scale:
        nm_per_px = estimate_nm_per_pixel_from_scale_bar(gray, 200.0)

    vis = draw_overlay(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), results, nm_per_px, max(1, args.line_thickness))
    cv2.imwrite(str(args.output), vis)

    n_in = sum(1 for _, _, r_in, _, _, _, _, _ in results if r_in is not None)
    n_out = sum(1 for _, _, _, r_out, _, _, _, _ in results if r_out is not None)
    print(f"중심 검출 개수: {len(results)}")
    print(f"내부 원 인식: {n_in} / {len(results)}")
    print(f"외곽 원 인식: {n_out} / {len(results)}")
    shifts = [(ocx - cx, ocy - cy) for cx, cy, _, r_out, _, _, ocx, ocy in results if r_out is not None and ocx is not None and ocy is not None]
    if shifts:
        dxs = np.array([s[0] for s in shifts], dtype=np.float32)
        dys = np.array([s[1] for s in shifts], dtype=np.float32)
        if nm_per_px is not None:
            print(f"shift 평균 dx={dxs.mean() * nm_per_px:.3f}nm, dy={dys.mean() * nm_per_px:.3f}nm")
            print(f"shift 절대평균 |dx|={np.mean(np.abs(dxs)) * nm_per_px:.3f}nm, |dy|={np.mean(np.abs(dys)) * nm_per_px:.3f}nm")
        else:
            print(f"shift 평균 dx={dxs.mean():.3f}px, dy={dys.mean():.3f}px")
            print(f"shift 절대평균 |dx|={np.mean(np.abs(dxs)):.3f}px, |dy|={np.mean(np.abs(dys)):.3f}px")
    if nm_per_px:
        print(f"보정: 약 {nm_per_px:.4f} nm/px")
    else:
        print("보정 없음: 둘레 단위는 px")
    print(f"결과 저장: {args.output.resolve()}")

    csv_path = args.output.with_suffix(".csv")
    if nm_per_px is None:
        raise SystemExit("CSV를 nm 단위로 저장하려면 스케일 보정이 필요합니다. --nm-per-pixel 값을 지정하세요.")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "inner_perimeter_nm",
                "outer_perimeter_nm",
                "shift_dx_nm",
                "shift_dy_nm",
            ]
        )
        for idx, (cx, cy, _, r_out, inner_perim_px, _, ocx, ocy) in enumerate(results, start=1):
            if ocx is None or ocy is None or r_out is None:
                continue
            dx_nm = (ocx - cx) * nm_per_px
            dy_nm = (ocy - cy) * nm_per_px
            inner_nm = inner_perim_px * nm_per_px
            out_perim_nm = (2.0 * math.pi * r_out) * nm_per_px
            w.writerow([idx, f"{inner_nm:.3f}", f"{out_perim_nm:.3f}", f"{dx_nm:.3f}", f"{dy_nm:.3f}"])
    print(f"시프트/둘레 표 저장: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
