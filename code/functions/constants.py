"""Shared constants for cell-type taxonomy, colors, and labels.

All subclass and supertype names are from the AIBS mouse V1 taxonomy.
Supertype colors are lightness-varied shades of their parent subclass color.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Mouse / session / stimulus constants
# ---------------------------------------------------------------------------
MOUSE_IDS = ['778174', '786297', '797371']
SESSIONS = ['session_1', 'session_2', 'session_3']
ORIENTATIONS = np.array([0, 45, 90, 135, 180, 225, 270, 315])
CONTRASTS = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
TFS = np.array([1, 2, 4, 8, 15])

# ---------------------------------------------------------------------------
# Subclass constants
# ---------------------------------------------------------------------------
SUBCLASS_ORDER = [
    '007 L2/3 IT CTX Glut',
    '006 L4/5 IT CTX Glut',
    '022 L5 ET CTX Glut',
    '052 Pvalb Gaba',
    '051 Pvalb chandelier Gaba',
    '053 Sst Gaba',
    '046 Vip Gaba',
    '047 Sncg Gaba',
    '049 Lamp5 Gaba',
]

SUBCLASS_COLORS = {
    '007 L2/3 IT CTX Glut':       '#1f77b4',
    '006 L4/5 IT CTX Glut':       '#2ca02c',
    '022 L5 ET CTX Glut':         '#9467bd',
    '052 Pvalb Gaba':             '#d62728',
    '051 Pvalb chandelier Gaba':  '#ff9896',
    '053 Sst Gaba':               '#ff7f0e',
    '046 Vip Gaba':               '#e377c2',
    '047 Sncg Gaba':              '#17becf',
    '049 Lamp5 Gaba':             '#8c564b',
}

SUBCLASS_SHORT = {
    '007 L2/3 IT CTX Glut':       'L2/3 IT',
    '006 L4/5 IT CTX Glut':       'L4/5 IT',
    '022 L5 ET CTX Glut':         'L5 ET',
    '052 Pvalb Gaba':             'Pvalb',
    '051 Pvalb chandelier Gaba':  'Pvalb ChC',
    '053 Sst Gaba':               'Sst',
    '046 Vip Gaba':               'Vip',
    '047 Sncg Gaba':              'Sncg',
    '049 Lamp5 Gaba':             'Lamp5',
}

# ---------------------------------------------------------------------------
# Supertype constants
# ---------------------------------------------------------------------------
SUPERTYPE_ORDER = [
    # L2/3 IT
    '0029 L2/3 IT CTX Glut_1',
    '0030 L2/3 IT CTX Glut_2',
    '0031 L2/3 IT CTX Glut_3',
    '0032 L2/3 IT CTX Glut_4',
    # L4/5 IT
    '0023 L4/5 IT CTX Glut_1',
    '0024 L4/5 IT CTX Glut_2',
    '0025 L4/5 IT CTX Glut_3',
    '0026 L4/5 IT CTX Glut_4',
    '0027 L4/5 IT CTX Glut_5',
    '0028 L4/5 IT CTX Glut_6',
    # L5 ET
    '0090 L5 ET CTX Glut_1',
    '0092 L5 ET CTX Glut_3',
    '0093 L5 ET CTX Glut_4',
    # Pvalb
    '0206 Pvalb Gaba_2',
    '0207 Pvalb Gaba_3',
    '0208 Pvalb Gaba_4',
    '0209 Pvalb Gaba_5',
    '0210 Pvalb Gaba_6',
    '0212 Pvalb Gaba_8',
    # Pvalb chandelier
    '0204 Pvalb chandelier Gaba_1',
    # Sst
    '0215 Sst Gaba_2',
    '0216 Sst Gaba_3',
    '0218 Sst Gaba_5',
    '0219 Sst Gaba_6',
    '0220 Sst Gaba_7',
    '0221 Sst Gaba_8',
    '0222 Sst Gaba_9',
    '0223 Sst Gaba_10',
    '0224 Sst Gaba_11',
    '0225 Sst Gaba_12',
    '0227 Sst Gaba_14',
    '0228 Sst Gaba_15',
    '0230 Sst Gaba_17',
    '0231 Sst Gaba_18',
    # Vip
    '0173 Vip Gaba_1',
    '0174 Vip Gaba_2',
    '0175 Vip Gaba_3',
    '0176 Vip Gaba_4',
    '0177 Vip Gaba_5',
    '0178 Vip Gaba_6',
    '0179 Vip Gaba_7',
    '0181 Vip Gaba_9',
    '0183 Vip Gaba_11',
    '0184 Vip Gaba_12',
    # Sncg
    '0185 Sncg Gaba_1',
    '0186 Sncg Gaba_2',
    '0187 Sncg Gaba_3',
    '0190 Sncg Gaba_6',
    '0192 Sncg Gaba_8',
    # Lamp5
    '0199 Lamp5 Gaba_1',
    '0200 Lamp5 Gaba_2',
    '0201 Lamp5 Gaba_3',
    '0202 Lamp5 Gaba_4',
]

SUPERTYPE_COLORS = {
    # L2/3 IT shades (blue)
    '0029 L2/3 IT CTX Glut_1': '#6db4e6',
    '0030 L2/3 IT CTX Glut_2': '#3196db',
    '0031 L2/3 IT CTX Glut_3': '#1d70a9',
    '0032 L2/3 IT CTX Glut_4': '#13486d',
    # L4/5 IT shades (green)
    '0023 L4/5 IT CTX Glut_1': '#73d973',
    '0024 L4/5 IT CTX Glut_2': '#53d053',
    '0025 L4/5 IT CTX Glut_3': '#36c436',
    '0026 L4/5 IT CTX Glut_4': '#2da42d',
    '0027 L4/5 IT CTX Glut_5': '#248424',
    '0028 L4/5 IT CTX Glut_6': '#1c641c',
    # L5 ET shades (purple)
    '0090 L5 ET CTX Glut_1': '#d3c0e4',
    '0092 L5 ET CTX Glut_3': '#9467bd',
    '0093 L5 ET CTX Glut_4': '#543273',
    # Pvalb shades (red)
    '0206 Pvalb Gaba_2': '#eb9192',
    '0207 Pvalb Gaba_3': '#e36767',
    '0208 Pvalb Gaba_4': '#db3c3d',
    '0209 Pvalb Gaba_5': '#c12324',
    '0210 Pvalb Gaba_6': '#971b1c',
    '0212 Pvalb Gaba_8': '#6c1414',
    # Pvalb chandelier (salmon)
    '0204 Pvalb chandelier Gaba_1': '#ff9896',
    # Sst shades (orange)
    '0215 Sst Gaba_2': '#ffc38e',
    '0216 Sst Gaba_3': '#ffb87a',
    '0218 Sst Gaba_5': '#ffae66',
    '0219 Sst Gaba_6': '#ffa353',
    '0220 Sst Gaba_7': '#ff993f',
    '0221 Sst Gaba_8': '#ff8f2b',
    '0222 Sst Gaba_9': '#ff8418',
    '0223 Sst Gaba_10': '#ff7a04',
    '0224 Sst Gaba_11': '#f07000',
    '0225 Sst Gaba_12': '#dc6700',
    '0227 Sst Gaba_14': '#c85e00',
    '0228 Sst Gaba_15': '#b55500',
    '0230 Sst Gaba_17': '#a14c00',
    '0231 Sst Gaba_18': '#8e4200',
    # Vip shades (pink)
    '0173 Vip Gaba_1': '#f2c0e3',
    '0174 Vip Gaba_2': '#eeacda',
    '0175 Vip Gaba_3': '#ea98d1',
    '0176 Vip Gaba_4': '#e684c8',
    '0177 Vip Gaba_5': '#e270bf',
    '0178 Vip Gaba_6': '#de5db6',
    '0179 Vip Gaba_7': '#d949ad',
    '0181 Vip Gaba_9': '#d535a4',
    '0183 Vip Gaba_11': '#c92998',
    '0184 Vip Gaba_12': '#b52589',
    # Sncg shades (cyan)
    '0185 Sncg Gaba_1': '#76e4f0',
    '0186 Sncg Gaba_2': '#42daea',
    '0187 Sncg Gaba_3': '#18c8da',
    '0190 Sncg Gaba_6': '#1399a6',
    '0192 Sncg Gaba_8': '#0d6973',
    # Lamp5 shades (brown)
    '0199 Lamp5 Gaba_1': '#c59b92',
    '0200 Lamp5 Gaba_2': '#ac7063',
    '0201 Lamp5 Gaba_3': '#825045',
    '0202 Lamp5 Gaba_4': '#53332c',
}

SUPERTYPE_SHORT = {
    # L2/3 IT
    '0029 L2/3 IT CTX Glut_1': 'L2/3 IT_1',
    '0030 L2/3 IT CTX Glut_2': 'L2/3 IT_2',
    '0031 L2/3 IT CTX Glut_3': 'L2/3 IT_3',
    '0032 L2/3 IT CTX Glut_4': 'L2/3 IT_4',
    # L4/5 IT
    '0023 L4/5 IT CTX Glut_1': 'L4/5 IT_1',
    '0024 L4/5 IT CTX Glut_2': 'L4/5 IT_2',
    '0025 L4/5 IT CTX Glut_3': 'L4/5 IT_3',
    '0026 L4/5 IT CTX Glut_4': 'L4/5 IT_4',
    '0027 L4/5 IT CTX Glut_5': 'L4/5 IT_5',
    '0028 L4/5 IT CTX Glut_6': 'L4/5 IT_6',
    # L5 ET
    '0090 L5 ET CTX Glut_1': 'L5 ET_1',
    '0092 L5 ET CTX Glut_3': 'L5 ET_3',
    '0093 L5 ET CTX Glut_4': 'L5 ET_4',
    # Pvalb
    '0206 Pvalb Gaba_2': 'Pvalb_2',
    '0207 Pvalb Gaba_3': 'Pvalb_3',
    '0208 Pvalb Gaba_4': 'Pvalb_4',
    '0209 Pvalb Gaba_5': 'Pvalb_5',
    '0210 Pvalb Gaba_6': 'Pvalb_6',
    '0212 Pvalb Gaba_8': 'Pvalb_8',
    # Pvalb chandelier
    '0204 Pvalb chandelier Gaba_1': 'Pvalb ChC_1',
    # Sst
    '0215 Sst Gaba_2': 'Sst_2',
    '0216 Sst Gaba_3': 'Sst_3',
    '0218 Sst Gaba_5': 'Sst_5',
    '0219 Sst Gaba_6': 'Sst_6',
    '0220 Sst Gaba_7': 'Sst_7',
    '0221 Sst Gaba_8': 'Sst_8',
    '0222 Sst Gaba_9': 'Sst_9',
    '0223 Sst Gaba_10': 'Sst_10',
    '0224 Sst Gaba_11': 'Sst_11',
    '0225 Sst Gaba_12': 'Sst_12',
    '0227 Sst Gaba_14': 'Sst_14',
    '0228 Sst Gaba_15': 'Sst_15',
    '0230 Sst Gaba_17': 'Sst_17',
    '0231 Sst Gaba_18': 'Sst_18',
    # Vip
    '0173 Vip Gaba_1': 'Vip_1',
    '0174 Vip Gaba_2': 'Vip_2',
    '0175 Vip Gaba_3': 'Vip_3',
    '0176 Vip Gaba_4': 'Vip_4',
    '0177 Vip Gaba_5': 'Vip_5',
    '0178 Vip Gaba_6': 'Vip_6',
    '0179 Vip Gaba_7': 'Vip_7',
    '0181 Vip Gaba_9': 'Vip_9',
    '0183 Vip Gaba_11': 'Vip_11',
    '0184 Vip Gaba_12': 'Vip_12',
    # Sncg
    '0185 Sncg Gaba_1': 'Sncg_1',
    '0186 Sncg Gaba_2': 'Sncg_2',
    '0187 Sncg Gaba_3': 'Sncg_3',
    '0190 Sncg Gaba_6': 'Sncg_6',
    '0192 Sncg Gaba_8': 'Sncg_8',
    # Lamp5
    '0199 Lamp5 Gaba_1': 'Lamp5_1',
    '0200 Lamp5 Gaba_2': 'Lamp5_2',
    '0201 Lamp5 Gaba_3': 'Lamp5_3',
    '0202 Lamp5 Gaba_4': 'Lamp5_4',
}

# Mapping: supertype_name -> subclass_name
SUPERTYPE_TO_SUBCLASS = {st: sc for sc in SUBCLASS_ORDER
                         for st in SUPERTYPE_ORDER
                         if SUBCLASS_SHORT[sc] in st or sc.split(' ', 1)[1].split(' Glut')[0].split(' Gaba')[0] in st}

# Build it explicitly for correctness
SUPERTYPE_TO_SUBCLASS = {}
_subclass_supertypes = {
    '007 L2/3 IT CTX Glut': [st for st in SUPERTYPE_ORDER if 'L2/3 IT CTX Glut' in st],
    '006 L4/5 IT CTX Glut': [st for st in SUPERTYPE_ORDER if 'L4/5 IT CTX Glut' in st],
    '022 L5 ET CTX Glut': [st for st in SUPERTYPE_ORDER if 'L5 ET CTX Glut' in st],
    '052 Pvalb Gaba': [st for st in SUPERTYPE_ORDER if 'Pvalb Gaba' in st and 'chandelier' not in st],
    '051 Pvalb chandelier Gaba': [st for st in SUPERTYPE_ORDER if 'Pvalb chandelier' in st],
    '053 Sst Gaba': [st for st in SUPERTYPE_ORDER if 'Sst Gaba' in st],
    '046 Vip Gaba': [st for st in SUPERTYPE_ORDER if 'Vip Gaba' in st],
    '047 Sncg Gaba': [st for st in SUPERTYPE_ORDER if 'Sncg Gaba' in st],
    '049 Lamp5 Gaba': [st for st in SUPERTYPE_ORDER if 'Lamp5 Gaba' in st],
}

SUBCLASS_SUPERTYPES = _subclass_supertypes

for sc, sts in _subclass_supertypes.items():
    for st in sts:
        SUPERTYPE_TO_SUBCLASS[st] = sc
