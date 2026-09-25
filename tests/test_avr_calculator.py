"""
Testes unitarios para o calculo cientifico do AVR (src/pipeline/avr_calculator.py).

Usa mascaras sinteticas (sem dataset, sem GPU) para validar:
1. A geometria do anel da Zona B (0.5-1.0 DD do disco optico).
2. A combinacao iterativa de Knudtson (maior+menor, nao fold-esquerda).
3. O pipeline completo calculate_scientific_avr com vasos de largura conhecida.
"""

import importlib.util
import math
import os

import cv2
import numpy as np

# Carrega avr_calculator.py diretamente pelo caminho do arquivo, sem passar
# pelo __init__.py de src.pipeline (que importa torch/torchvision para o
# resto do pipeline) -- este modulo e teste sao puros numpy/cv2/scipy/skimage
# e nao devem depender de uma instalacao de torch/torchvision valida.
_MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "pipeline", "avr_calculator.py")
)
_spec = importlib.util.spec_from_file_location("avr_calculator", _MODULE_PATH)
avr_calculator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(avr_calculator)

ScientificAVRCalculator = avr_calculator.ScientificAVRCalculator
CRAE_COEFFICIENT = avr_calculator.CRAE_COEFFICIENT
CRVE_COEFFICIENT = avr_calculator.CRVE_COEFFICIENT


def test_zone_b_ring_geometry():
    """A ROI deve incluir só o anel entre 0.5 DD e 1.0 DD do centro do disco."""
    shape = (400, 400)
    center = (200, 200)
    radius = 40.0  # raio do disco -> DD = 80px; zona B = [40px, 80px]
    calc = ScientificAVRCalculator(shape, optic_disc_center=center, optic_disc_radius=radius)
    roi = calc._create_roi_mask()

    # Dentro do disco (bem abaixo de 0.5 DD): fora da ROI.
    assert roi[200, 210] == 0, "pixel dentro do disco não deveria estar na Zona B"
    # Na metade da Zona B (~60px do centro): dentro da ROI.
    assert roi[200, 260] == 1, "pixel no meio da Zona B deveria estar incluído"
    # Bem além de 1.0 DD: fora da ROI.
    assert roi[200, 320] == 0, "pixel além de 1.0 DD não deveria estar na Zona B"
    print("[OK] geometria da Zona B (anel 0.5-1.0 DD)")


def test_knudtson_combination_uses_largest_smallest_pairing():
    """
    A combinação deve parear MAIOR com MENOR a cada passo, não um fold da
    esquerda ingênuo (bug do notebook original).
    """
    widths = [10.0, 8.0, 6.0, 4.0]

    # Fold ingênuo (bug antigo): sempre index 0 + index 1, reinserido na frente.
    naive = widths.copy()
    while len(naive) > 1:
        w1, w2 = naive[0], naive[1]
        naive = [CRAE_COEFFICIENT * math.sqrt(w1 ** 2 + w2 ** 2)] + naive[2:]
    naive_result = naive[0]

    # Algoritmo canônico: parear maior+menor, reordenar, repetir.
    result = ScientificAVRCalculator._combine_knudtson(widths, CRAE_COEFFICIENT)

    # Cálculo manual do canônico para [4,6,8,10]:
    # passo 1: menor=4, maior=10 -> c1 = 0.88*sqrt(10^2+4^2); resto ordenado [6,8,c1]
    c1 = CRAE_COEFFICIENT * math.sqrt(10.0 ** 2 + 4.0 ** 2)
    remaining = sorted([6.0, 8.0, c1])
    # passo 2: menor e maior de remaining
    smallest, largest = remaining[0], remaining[-1]
    mid = [w for w in remaining if w not in (smallest, largest)] or [remaining[1]]
    c2 = CRAE_COEFFICIENT * math.sqrt(largest ** 2 + smallest ** 2)
    expected = CRAE_COEFFICIENT * math.sqrt(max(c2, mid[0]) ** 2 + min(c2, mid[0]) ** 2)

    assert abs(result - expected) < 1e-6, f"esperado {expected}, obtido {result}"
    assert abs(result - naive_result) > 1e-6, (
        "resultado canônico não deveria coincidir com o fold ingênuo neste caso"
    )
    print(f"[OK] combinação Knudtson maior-menor (canônico={result:.4f}, fold ingênuo={naive_result:.4f})")


def test_knudtson_single_and_empty_width():
    assert ScientificAVRCalculator._combine_knudtson([], CRAE_COEFFICIENT) == 0.0
    assert ScientificAVRCalculator._combine_knudtson([5.0], CRAE_COEFFICIENT) == 5.0
    print("[OK] casos-base (lista vazia / único calibre)")


def _draw_vessels(shape, center, count, angle_offset_deg, length, thickness):
    """Desenha `count` linhas retas saindo do centro, simulando vasos radiais."""
    mask = np.zeros(shape, dtype=np.uint8)
    for i in range(count):
        angle = math.radians(angle_offset_deg + i * (360.0 / count))
        x0 = int(center[0] + 20 * math.cos(angle))
        y0 = int(center[1] + 20 * math.sin(angle))
        x1 = int(center[0] + length * math.cos(angle))
        y1 = int(center[1] + length * math.sin(angle))
        cv2.line(mask, (x0, y0), (x1, y1), color=255, thickness=thickness)
    return mask


def test_calculate_scientific_avr_with_known_widths():
    """
    Desenha 6 'artérias' radiais mais finas e 6 'veias' radiais mais grossas
    cruzando a Zona B, e confere que CRAE < CRVE e que o AVR calculado fica
    perto da razão de larguras nominal (a razão exata de Knudtson difere de
    largura-bruta/largura-bruta, então validamos só a direção e a ordem de
    grandeza).
    """
    shape = (600, 600)
    center = (300, 300)
    radius = 40.0  # zona B = [40px, 80px] do centro

    artery_thickness = 6   # px
    vein_thickness = 10    # px

    artery_mask = _draw_vessels(shape, center, count=6, angle_offset_deg=0, length=120, thickness=artery_thickness)
    vein_mask = _draw_vessels(shape, center, count=6, angle_offset_deg=30, length=120, thickness=vein_thickness)

    calc = ScientificAVRCalculator(shape, optic_disc_center=center, optic_disc_radius=radius)
    result = calc.calculate_scientific_avr(artery_mask, vein_mask)

    assert result["status"] == "SUCCESS", result
    assert result["crae"] > 0 and result["crve"] > 0
    assert result["crae"] < result["crve"], "artérias mais finas deveriam gerar CRAE < CRVE"
    assert 0.0 < result["avr"] < 1.0
    assert result["measurements"]["artery_count"] >= 2
    assert result["measurements"]["vein_count"] >= 2
    print(f"[OK] pipeline completo: AVR={result['avr']:.3f} CRAE={result['crae']:.2f} CRVE={result['crve']:.2f} "
          f"risco={result['risk_level']}")


def test_insufficient_vessels():
    """Sem vasos suficientes na Zona B, deve retornar INSUFFICIENT_VESSELS, não quebrar."""
    shape = (300, 300)
    empty_mask = np.zeros(shape, dtype=np.uint8)
    calc = ScientificAVRCalculator(shape, optic_disc_center=(150, 150), optic_disc_radius=20.0)
    result = calc.calculate_scientific_avr(empty_mask, empty_mask)
    assert result["status"] == "INSUFFICIENT_VESSELS"
    assert result["avr"] == 0.0
    print("[OK] fallback de vasos insuficientes")


def test_fallback_center_without_real_optic_disc():
    """Sem od_center/od_radius reais, ainda funciona (fallback), mas não deve crashar."""
    shape = (400, 400)
    mask = _draw_vessels(shape, (200, 200), count=6, angle_offset_deg=0, length=150, thickness=8)
    calc = ScientificAVRCalculator(shape)  # sem od_center/od_radius -> fallback com warning
    assert calc.od_center == (200, 200)
    assert calc.od_radius == min(shape) * 0.15
    result = calc.calculate_scientific_avr(mask, mask)
    assert result["status"] in ("SUCCESS", "INSUFFICIENT_VESSELS")
    print("[OK] fallback sem disco óptico real não quebra (mas usa heurística documentada)")


if __name__ == "__main__":
    try:
        test_zone_b_ring_geometry()
        test_knudtson_combination_uses_largest_smallest_pairing()
        test_knudtson_single_and_empty_width()
        test_calculate_scientific_avr_with_known_widths()
        test_insufficient_vessels()
        test_fallback_center_without_real_optic_disc()
        print("\nTodos os testes do AVR calculator passaram.")
    except Exception as e:
        print(f"\nTeste falhou: {e}")
        raise
