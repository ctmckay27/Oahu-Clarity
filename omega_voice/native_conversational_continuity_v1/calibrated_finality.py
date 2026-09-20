"""Measured Mari native finality/continuation direction, compact projection.

The payload is a per-layer int8 quantization of the single calibrated direction
from MARI_ISOLATED_NATIVE_CONTROL_EVIDENCE_2026-09-19.zip. It is not an emotion
vector or a generic style control. Quantization preserves cosine similarity
above 0.9997 to the measured float32 direction; the original evidence and proof
ceiling remain authoritative.
"""
from __future__ import annotations
import base64,hashlib,zlib
import numpy as np

SOURCE_ARCHIVE_SHA256 = "6f520a4937efdc1fb98fd04f956d144f999237aa4f02cfce4e41a5045d042f48"
SOURCE_NPZ_SHA256 = "f6fb02e3dd7117045792eb681d0f07671c2203912111b36c1ebc0d26f3ade748"
SOURCE_DIRECTION_RAW_SHA256 = "7759fd94fe6860494972397236d76d17636dc8124096aadb82003cfcc89a8fd2"
QUANTIZED_PAYLOAD_SHA256 = "9779e548931fb6d5111d34f82a1c20c41595662ad1a6ac082946719f3446d4c0"
COSINE_TO_SOURCE = 0.9997504353523254
RELATIVE_L2_ERROR = 0.022345775738358498
SHAPE = (29, 2048)
_COMPRESSED_B64 = (
    "eNrtmkmvrdl511ffvN1uT3O7KruqjJ2E2I5F4jQVbKQghUSKIBJigBgkQso3AImJkRgwY8AIiSEDT4AoJEhkQCIYQGyICcaJXWWX"
    "q+r2p9nN269+8dwhcyag9yfdwb13n92s9Tz/Zusg9I+/hv4v//m7//Dvv/9X0L97/5/94cv3d7/Ofvn3/1H1y//nYxYWFhYWFhYW"
    "FhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYW"
    "FhYWFhYWFhb+32NzGG2B8LwztUYan1NboCEwtI2pPQzqlviDvPdth4qCppSmqzMWF0kkNPXs66PsmpvLXZE2mkVhj9u5/KJld8NU"
    "T6v1fDJkTmEeVmO3+V67M9LqxvtzKElDa8r2nCVGj+RZFqUVOJ+e/WifBoo6kb+V8ReEWpOxLV9OR1pdfPf2aO1tEV1ONbqRUfXx"
    "5Lqa8zY8UfLBp/60wdefCH9Yf/Ir4ambLmwQx81cD52qpgONW/qEbfbjMbmQ+Do3vF6zTnqypTnQl/nBdSjwwdFnr/GwnwaytbQn"
    "JmaCd7I2ev74LhaY7XqphrOgMfat+quOHaZ3JD2jJ7cjSy3G8zls+wobnHfXH0fubUgtu+PD9NkXj3H7zYAs90GH+z609r0qV7Le"
    "360DrtKeqqqzRG3SKh9sv2qx7kIhhS3+60BfGMHgPR5w7oVu8gFOXq0qEn2a6wmZYhDctLFoxrU4Of7iTmqjVMy70V+eX/Iy/vFI"
    "n1j9vTVDlKP1xjQvN1cpds00rFfyYRLqMxZ9/MRN8009+zF2/gsqo5Q0hqtB1cDVaE432lf80TZuTC8RlvtBSJoG29mTGWsYHo1Z"
    "vigFYk03idWTkygyGmQ3nEfD18/Yb10Px9vQjKJipyyVIUomWQUxqGnsQ7dDLfZD3pUHXN+nsLNPaDIjoRtHsa3ezQMJtm8luteC"
    "siSmNSo+Ho+xYsP0R8k+Pj9/mFfvMurOVc+PxXfju0/pUU1v06T7sM2TlrWrT3l8mfH5g1sz/uq6aKVwLyO3bKXXrxqljNgFedHl"
    "+VlF4/NKzzrFnhnWlj8eov8nyNmeEeHyKJhIfH+KCRm1Ww+UsZ97+OnhpLsvbA49Hj81SpLOtRQNs6/3u/rrFd/PuYKF2XylyxfU"
    "dP3Q0AmtDoTIOg+hi6eLKJLAkVCy+weYBXN7hYqpwG+hajbZjB+v0NDiq9mxMlQ51i3L39qjkEl5M4ykzY+ZiheW714P7BN6SkiF"
    "iKLa23KL9Uy3olQ+5Ol/bJ5yujnpkt973ajdNugdfkk3toi2z32/bY55S65ur9ZfEvItrtLx5qYMXWBPHUK55ejnVitGeWR/001+"
    "jV7BSrnpK/Ov8hIOmRiUEJ/yFZuO871h4bqg4nSmu+iGl+2ruj0Vq83qfjKnQm74Vs89QUVyNz6n44+HI7Mc989tdYXLQ5+vkRG/"
    "1PqWv3Xy0ql5Qi/XPalMcW84Sg/zDJ9k1yPRVoK210eyStxR8V7AO7ihxPQV2gpZnNz5dYGHsZiw7cmsSmRysp9oRory2B6Hr5fK"
    "+axxsX27vTNZ5NvxrqVzr/svvs3T9l0ZeYcTJdtUh9W9+hE95I3yIqTiZeVHOaZXf6C7+7WoSYiqzvP5bnXKv7JD27X8sSxu/Ifr"
    "9UlP533bbAo62M9+nx3D7vC6zi+mmIuNzDX3jfq1wYs283rYPKs5Ckfzrrbnb4SVbxhsLsm3f/7ifPFFH0wInxkcWonv+u08klWh"
    "Kzw/9Sr4QxvltWCDKlUsPvqskdtXsiZM+a+NsIPxljXFKW+4+TmsnrjL+PZBN5/u6Y/E23O7vggVvnq5J9P2Ye7+iOMVtVfFpbcE"
    "609NHH0rSxi7b86m53+Hfr4df1c2j6P3JFTaf1Rgy0wot2kIGWO8fuc03KPNqts63lq9c1Slm+4oCj68cD18YtSf7nYOrVkcZX6e"
    "6t0PsDtSYTaajr1GN/u59rlSc8sUed7cZSM6vL6NiG7ka7luWPHgRyD9j+UL/qUH5votqnZ0XXjQoop1YEUrp1+L+YGsbsUXfrLq"
    "9mhUDCdR/fYspuZpv2VnczweiCiIgtG3s362efr4k3brJ1LczbelSGklzvj4SVIkN+Hztpq/M9zNVde6x1aQhBhHomv8uY0s0uH1"
    "eT9vHl3YGbXrguUZIYpneuENHrj7jlK+emefXGvSSBMcmbwplf3ji674k7P1+9Nsv8ISXpFgGstcAz8/5edFc3G60O88xWc8j5+G"
    "bM1ENzSvlAW58Uz+x7Ha9M5tzjG5WRV9vJ8LGo91/VwosV1ZEv/pc+1p9aQJ8tba6TCHjp9+naJofXX3MPmb/foahhiNGyIp8YoV"
    "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
ÜÑ\[Þ™”šÎMQ™]•šZš][ÙL^–YÒQL”SJÍŠÛQœT–TŠÖMˆ‚ˆ›ZYŽ‘’LLÔ’Õ^žRLTLœ›ÜÙÕ™VMŽQÚ]›Lš’–œZ“˜ÊÝÔ™ÚÖYžTSÕ‹ÌÞ“Ñ’RŽÕÛŽQZ›PÝšS”ËÐÒÙY™™‚ˆURÜZ]Ôœ–]QUQUKÒÓÑÒÙZ’ÕšœSZÔR›ÊØÔÐT’žLÞ›[SÝ›“XØÊÒT\ÕQPÚš™XÞ•V›ÑM˜›LØÖ”È‚ˆ•ÐÐN]X“˜UVÓÕKÓ›ÓÖZÜUPÎLRÐ‘]KÕŽV›’QMPÙYÓœÙ’RÐ\ž›•^›QÙ]SXÓ\œÝTÚMÞUšPH‚ˆŒšMYŒÜ•ÚÕÍ^ÞÑÝM•ÔšœRÜ”‹ÛL–˜”ÙPÑØ™š›LS‘UT˜Z“Ò]Ô˜Œ•\•RÍPÚŠÚ“PÍÙÖ‘UÚZÐ[Rˆ‚ˆ”SŒÚ“ÍM^‘ŒÒTTšÔ•Y“TMœŽ’ÕÖZ›V^“šZÐÓ•KÞ•ÊÓÝ‘œ^MÓÍ›\RÚQ^U˜ÌŽYVÛL“•Ð^RL\ÜNPÖXQÎPKÒ•Ý“™”ˆ‚ˆššÌ
ÓÚ”•Õš•ÙÔ]ÍPÍZ\QÊÎURSJÔ‹ÊÚÔ‘ÍÐ]“ŒÌQ–Z\“MÕÕÔSœV“ÓÓSVž^QTÝLØT”ÑØ^Y™‚ˆŒš•Ž[‹ÖZYŽÔSÎ[ÛÝ]ŒÎÊÑØ•Í›ÎYÐLÊÞYÝÞMÌÓÒKØÔ]ÊÕ••JÑ]’UXSËÛMUÙÜÍU”œUR[•PÛšXVXÓ‚ˆ‘Œ˜YÑTÐØšLPTMJÜÌ”S^Þ^YQÐ›ÍÎUMP^ØSÊÜQÑÎ›‘S[Q\TÙÍÕRZÝÙZÍÑM^]S]ÚVRÔTLZØÚš‚ˆŠÖUÙ™L\ÓÑQ–œ•TÌŽ[Õ“S^Ü\YMÛœLÕÜNYÜ”‘ÜÑ\TžYZSž\ÎÜŒÊÊÔ‘Ú‘ÝÔSLœŒ“ÛÍQÞ\œZÜ™“ÜÌL‚ˆŒZÖšÊÔXØœÍ”™ØŒÛš“ÝÜYÒQU˜Ö[šÌ˜ÜËÝ–‘PÔÌÓÞ‘ØÝÌÝT–œUNPÕMP[Üš””ÑŽKÓUÓžR“PZRU™›Z‘ÌÖ“H‚ˆ”ÜÍMÖ˜PS•ÚÞÓÝL›PQZÒXÚZÑÚžœJÐÛSÞÎ[ÓPQÌQÜMTÛÐØ]ØÙÜ˜’R‘Z˜SÖVXÍZQ•ÝœÓ“ž‘ZÓRÖŠÔ^ZÒÈ‚ˆ’]ÞÔ”SÙYÕQÛM“Ì’Ì˜ÍZØU\[MÛ’ŒœÒQXÓ’™U‹ÝÎ^U“Òœ’TÜ”ÛÚ]ZÝœ™ØÜÙŽÑ‘“VÔÙQRUÖS\•˜H‚ˆS”žLXSœÖÜÓQÝÕ“ÙÝËÑÛ‘JÓT”ÚÝTœÜÕÛU“ÕQJÞU[’ÕÔ™ÒÍLÝ–™ÌÒM••Ø•šž˜ÛÝ\•ÕRV™XÚÜ[QšH‚ˆŒÝÕÞš^^MÛP\RÑŒ–N™QÍØÚ‘ÌÚY•UÚÛšRÔž›ŽMšQ˜\žÜRÌ]ÔUV“YÜ‘›œ›ŒÖš’›˜–U’œÛ“ÝJÍÓŽNZ›SÈ‚ˆ•L‘UžQÐ‘MÔ[“”UÚV˜ÓÜ˜YÔLžÖQLNQTžØÑ\˜NX–‘ÊÖÛÚTÓZØR[Nž‘Ý[T›NV››RMÍ•”ÎTÓÓÚ‘Úž’ˆ‚ˆ•ØÒ^
ÚÎœ›ž’‘ÚSÕš™ÕÒVž“ÛÝZŠÜ’[LÚÓ–œRš[ÑšŒÙ“ÐVS•ØØX\ÕRÜZX”Uš˜Ù\ŠÞS[]LÍQ”È‚ˆš\”Ò”MQXÎX˜“LLÑÕÒÒX››T•ÝŽ–›Ý]œMLZL‘žÕš]ŠÔÒÖ[ÓÝ][Õ‘˜QÝ[ÙÔÑÑÙ’Öœ\ÓÕÎMØRS\žTRYÈ‚ˆQÒÍ\T™˜]ØÕšœ“œÕÙÜÓÌÛŠÜš’ÚØÖ•œžŒÌÔMQ™™V˜ÕMXÙ‘“ÞYYÍÌ\ÖMUS\ÜšRY’UÞ›šT”ÙÚVH‚ˆ˜]ÜÊÕ]TQÐUÑYLQVœ\JÕPšÞQÛÛØÐÛ”™›ÜÚSLÎR‹ÙRÑZV”•ÐÕÔÜ–^QT’“ÍT–’Y›ÝÊÚžXÖŽTØ\ÐÝŠÒ™ZT‚ˆŒXÎÍ‹Û›TY[Z^QÑRTÎ\Óœ[ÊÊÒSØ”ÜÌÌž•ÕÒÕÝËÍM]ÞV™‘”T^š”ÑZŒ›RÚKÐKÞY•”•UY]Ú”ÓÓM™Y‘“È‚ˆšX“ÙÔ“ÑZRÜÖ[“MRŒTP›Î”KÚž‘UXœ”ÔTZ]LQÛR\^MY•[NÜØMžTÒÌÜÕX“^•Ô[^ž˜›•^T
ÎZÜÝÑÍH‚ˆœÓÝÜÓÝÜÓÝÜÓÝÜÓÝÜÓÝÜÓÝÎ‹ÞRL]Yˆ‚ŠB‚™YˆØYÙ\™XÝ[ÛŠ
HOˆœ›™\œ˜^N‚ˆ˜]Ï^›X‹™XÛÛ\™\ÜÊ˜\ÙM˜XÛÙJÐÓÓT‘TÔÑQÐ
JBˆYˆ\ÚX‹œÚLMŠ˜]ÊKš^YÙ\Ý

HOTUPS•V‘QÔVSÐQÔÒLMŽ‚ˆ˜Z\ÙH[[YQ\œ›ÜŠ˜Ø[Xœ˜]Y\™XÝ[Ûˆ^[ØY˜Z[Y[YÜš]HÚXÚÈŠBˆØØ[\Ï[œ™œ›ÛXY™™\Š˜]ÖÎŒŽJK\OHŠK˜ÛÜJ
BˆO[œ™œ›ÛXY™™\Š˜]ÖÌŽJ—K\O[œš[
K˜ÛÜJ
Kœ™\Ú\JÒTJBˆ\œ\K˜\Ý\Jœ™›Ø]ÌŠJœØØ[\ÖÎ‹›Û™WBˆYˆ›Ýœš\Ùš[š]J\œŠK˜[

HÜˆ\œ‹œÚ\HOTÒTN‚ˆ˜Z\ÙH[[YQ\œ›ÜŠš[˜[YØ[Xœ˜]Y\™XÝ[Ûˆ™XÛÛœÝXÝ[ÛˆŠBˆ™]\›ˆ\œ–Ó›Û™WB