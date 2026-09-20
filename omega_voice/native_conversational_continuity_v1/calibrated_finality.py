"""Measured Mari native finality/continuation direction, compact rank-4 projection.

Derived from MARI_ISOLATED_NATIVE_CONTROL_EVIDENCE_2026-09-19.zip.
This is a same-speaker boundary-contour actuator, not an emotion/style vector.
The original evidence remains authoritative; compact projection cosine > .9959.
"""
from __future__ import annotations
import base64,hashlib,zlib
import numpy as np

SOURCE_ARCHIVE_SHA256="6f520a4937efdc1fb98fd04f956d144f999237aa4f02cfce4e41a5045d042f48"
SOURCE_NPZ_SHA256="f6fb02e3dd7117045792eb681d0f07671c2203912111b36c1ebc0d26f3ade748"
SOURCE_DIRECTION_RAW_SHA256="7759fd94fe6860494972397236d76d17636dc8124096aadb82003cfcc89a8fd2"
COMPACT_RAW_SHA256="d402b7dcd6e3273e6960223eb0e89f81f676d01514bd65478d857e71d9b03af6"
COSINE_TO_SOURCE=0.9959902167320251
RELATIVE_L2_ERROR=0.08946140110492706
_COMPRESSED_B64=(
    "eNoNl1eMZNl53286N6e6lXPnnp7pnunZnp20aRh2l0uRWhKgIEsgJBiQAPvF9oNgwIANQi8GDMsPfLEBCzRgWzIpUWIQRXLz7sxy"
    "Z3Z74k7q3NXd1ZXTrZvjOb5vVQ9V5zvn+3+////7aYm/ub77/o3m3337xn97/X9+XPy337l5mvuPN3j233z8Vyv8jZfL/+nmhzM7"
    "rz35i7VPbv6HH71y9v3/cvPKH4FP9v/1f/34+z+Y+eSHmz+8+dml//vJX//V7I1zr/39azMf96734N9cJ64uX1f/9uXrEaAIk+A5"
    "h4wsBhkcLoQcjQWkzbNsLGFSRAQ0E1EM9CKcwiCHKYzv4jjEOS5oEyE7FLasGE0JGEElQ0XeEEQURUW44ZJs1AMcHYEIiORZ4Qgw"
    "TIggJjJOCOOQJKYkhnCWdNOBi4M0sKk8g3gmwn0qYlPYxI91jMhhPEe7hL2UTlEuNB0rDD1aCrHYDbkADwFH4AFidNnEIaaLEZY2"
    "xLYEWBgAPCYdYIhchHgxglN4aDseLZMhIPAo8iLow6yHM/0YZwgGD0yMQCQVqUwMTRrRY5wQEOEHYUBGiLClkuRbMWEDETKcHeMA"
    "kdb9WElFYyeuEMa2hOyQZQmcpyOBHoTTYZqIZBJZgMjjkHXEEAlzOLliYaxPURKiI8enHRd3zgU0jeSA9k1AgTAmJgQBaU8jUBi4"
    "MSDCywIuijHleZoAQRBScbDA+dALLCImCExnIBFoUUwxWIiYgFDZKSd4mIQz+MhjSCRkvVj6LhM3YPSiC+MghtANo9y4x4aQ8DJe"
    "GI1TGGNTPHcqxhEXkj4pYdQdCiMo1vQwioA6iGiOTIk+a3h9z/YI0YAOnMYIkgQvZCmRBDTlIoJ0/diDJApxYEwgjUBEIEgTNGCx"
    "vhj/L0vluMhn4yDiIhKAAAOADFIBQ5MERZMeEBDDUIHHYUEFl3wUIp4KaBjHFE46NuJxLKIFN8QNhCFRhwSLchIVhhHH/h5gDKTG"
    "lG94HE5jrCdYaW6qRpkh0/OYkAMU5vieAz0+ZFUepytMYN4KILBCCgMRhgwvyFgOJJgJIqMY0TxiWGkMCAwjqZBGKFwmEfkvk3bn"
    "cT6pyAkijwQuRyCPJHSLZ2Psbv9cAbpoGmRYyAIu8Gjg8RGtcpEbmX43DAGZDBPUu/cx5BA4o6AgJiiPxY3ACgQcQWESUZCMJITj"
    "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"
    "2S0lvpVTNN)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"
    "RjkMfA38/JR+ZdYLNrjMe3P0f5/99gF5gXVfIAKl90PBhU+X7sELDZericUPV3NfrcWuXwV7czsrj3q31nuwWU3/w5HxwhmdWjuv"
    "rr600JqpX+DO+tVHFPXZ17V05hPBBdoJONPClh71ps7j4WH1xQyxM7V/m2v2f1mGT6++e1DPkqfdPndrtLcQ1Pdj6qh3Dd/NRELf"
    "fnZ6RyTBTd6c5es55WAuTDFePu/Rh5NqOVrfF6Nb5I+1PV6dK+yfWuGWIRXenFAkBy+YYjnP/bkTOrnGB5UWk/5Ay/OdTUgstu/M"
    "yBcCeX6L2z9SmFJ0xpgQS+fxp/SPXl8gB0O5VPngbvF3/EY2JkqsM7vU05Wma3bEzdbs6oXUp+sfCbOp/7GGa5vmg8v2BVyTKefK"
    "7mJ4aVJuD+eoy+2no05zPP6T91qp2mDDIvvkYn7YQHpqH1u4/hwoJFXltsufXrpyRx2parN/9tmAGC4/mfauKz+20TM2fVF8GGeU"
    "ZcNa+l64PZcZZKOx197cH107sz5D6W89/oMNpiCPt1hJ++fx1taYKY6MT4rD1plnRzeEPUl+mdtt/W8Mm9+8+5XyucxRmEJn1wxI"
    "NbRJaSx8kPIm1Ap0buRm+7ioVazrJPu5Hv774bpafoHrDPd+Yo5rjxaF+pwERySnSmviaplnzxvSPx9Zq4Ywe17gjx70z93DtlZ6"
    "3Fv4hbXcV65daMXfuG3qqc/EM/DT99+7KWWp3OpyTAnWDFa+dH+cn1BXuPPiQTZd2Dn+sVtsaSnjSzaur9+6dUnI79450TPpQua+"
    "82w1ZaslH9sITu+zF9pTMkgX2WQLuXSdnrnkIo0fvq1dOir8P1haflLe23aa4XjY/YjtugFV0PjFwdn6YSwt3PK+gj7RqG9Usi/R"
    "P285y2X8lN2/VjCuBg8Oz98YDT/5vsSG3wh+ZkpZeJn8hbrYMLNuWjpKv7pXqB6+lc823719jjvZevUwSD3upek+9Q/B7nPd5euv"
    "gUaheVs/efHpyie5or9Vdzs+RY02HN18+3RrFiPrfZlSejd4ojVvUxvGxetvruxcBlRubNpYEYt3Z6sAkXpjU1f9Ct37ptSeHTnN"
    "7t3FG6nuHLE62lxOzbc3Z1e3xPb273IwvPCu82WrMGxH0B+TwfFhSf75yiq1WzHT/OBQH3Dew8n2pcqfdnrVs0wJv2GDC4vSAzw/"
    "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"
    "vROHSOH1ERTN6f8H0QC6wg=="
)

def load_direction()->np.ndarray:
    raw=zlib.decompress(base64.b64decode(_COMPRESSED_B64))
    if hashlib.sha256(raw).hexdigest()!=COMPACT_RAW_SHA256:
        raise RuntimeError("calibrated direction compact payload failed integrity check")
    a_bytes=5*4*4
    s_bytes=4*4
    a=np.frombuffer(raw[:a_bytes],dtype="<f4").copy().reshape(5,4)
    scales=np.frombuffer(raw[a_bytes:a_bytes+s_bytes],dtype="<f4").copy()
    q=np.frombuffer(raw[a_bytes+s_bytes:],dtype=np.int8).copy().reshape(4,2048)
    active=a@(q.astype(np.float32)*scales[:,None])
    out=np.zeros((1,29,2048),dtype=np.float32)
    out[0,21:26]=active
    if not np.isfinite(out).all():
        raise RuntimeError("invalid calibrated direction reconstruction")
    return out
