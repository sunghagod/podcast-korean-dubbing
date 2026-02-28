"""
podcast_icon.ico 생성 스크립트
"""
from PIL import Image, ImageDraw

def make_icon(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    s = size / 256  # 스케일

    # ── 배경 원 ──────────────────────────────────────────────
    m1 = max(1, int(2  * s))
    m2 = max(1, int(4  * s))
    m3 = max(1, int(7  * s))
    m4 = max(1, int(10 * s))

    # 글로우
    draw.ellipse([m1, m1, size-m1, size-m1], fill=(124, 58, 237, 35))
    # 배경 레이어
    draw.ellipse([m2, m2, size-m2, size-m2], fill=(12, 8, 28, 255))
    draw.ellipse([m3, m3, size-m3, size-m3], fill=(20, 12, 45, 255))
    draw.ellipse([m4, m4, size-m4, size-m4], fill=(30, 18, 65, 255))
    # 테두리 링
    draw.ellipse([m2, m2, size-m2, size-m2],
                 outline=(124, 58, 237, 220), width=max(1, int(3*s)))

    # ── 마이크 몸체 ───────────────────────────────────────────
    mw = int(50 * s)
    mh = int(70 * s)
    mx = cx - mw // 2
    my = cy - mh // 2 - int(18 * s)
    r  = int(25 * s)

    draw.rounded_rectangle(
        [mx, my, mx + mw, my + mh],
        radius=r,
        fill=(196, 181, 253, 255),
        outline=(167, 139, 250, 255),
        width=max(1, int(2*s)),
    )

    # 마이크 그릴 선
    for i in range(3):
        gy = my + int(20*s) + i * int(13*s)
        draw.line(
            [mx + int(9*s), gy, mx + mw - int(9*s), gy],
            fill=(109, 40, 217, 180),
            width=max(1, int(2*s)),
        )

    # ── 스탠드 (호 + 기둥 + 받침) ─────────────────────────────
    arm_r  = int(36 * s)
    arm_cy = my + mh + int(2 * s)
    lw = max(3, int(5 * s))

    # 호
    draw.arc(
        [cx - arm_r, arm_cy - arm_r, cx + arm_r, arm_cy + arm_r],
        start=0, end=180,
        fill=(196, 181, 253, 255),
        width=lw,
    )

    # 기둥
    pole_top = arm_cy + int(2 * s)
    pole_bot = cy + int(58 * s)
    draw.line([cx, pole_top, cx, pole_bot], fill=(196, 181, 253, 255), width=lw)

    # 받침
    bw = int(38 * s)
    draw.line([cx - bw, pole_bot, cx + bw, pole_bot], fill=(196, 181, 253, 255), width=lw)

    # ── 음파 (양쪽 호) ────────────────────────────────────────
    wave_cy = cy - int(8 * s)
    for rad, alpha in [(62, 200), (82, 130), (102, 70)]:
        r2 = int(rad * s)
        box = [cx - r2, wave_cy - r2, cx + r2, wave_cy + r2]
        w2  = max(2, int(4 * s))
        draw.arc(box, start=-58, end=58,  fill=(139, 92, 246, alpha), width=w2)
        draw.arc(box, start=122, end=238, fill=(139, 92, 246, alpha), width=w2)

    return img


def main():
    sizes  = [256, 128, 64, 48, 32, 16]
    images = [make_icon(s) for s in sizes]

    out = r"C:\Users\sungh\OneDrive\Desktop\팟케스트 번역\podcast_icon.ico"
    images[0].save(
        out,
        format="ICO",
        append_images=images[1:],
        sizes=[(s, s) for s in sizes],
    )
    print(f"✅ 아이콘 생성 완료: {out}")


if __name__ == "__main__":
    main()
