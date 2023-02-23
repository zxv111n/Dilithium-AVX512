#include <stdint.h>
#include "params.h"
#include "consts.h"

#define QINV 58728449 // q^(-1) mod 2^32
#define MONT -4186625 // 2^32 mod q
#define DIV 41978 // mont^2/256
#define DIV_QINV -8395782







const qdata_t qdata = {{
#define _16XQ 0
  Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q,

#define _16XQINV 16
  QINV, QINV, QINV, QINV, QINV, QINV, QINV, QINV, QINV, QINV, QINV, QINV, QINV, QINV, QINV, QINV,

#define _16XDIV_QINV 32
  DIV_QINV, DIV_QINV, DIV_QINV, DIV_QINV, DIV_QINV, DIV_QINV, DIV_QINV, DIV_QINV,
  DIV_QINV, DIV_QINV, DIV_QINV, DIV_QINV, DIV_QINV, DIV_QINV, DIV_QINV, DIV_QINV,

#define _16XDIV 48
  DIV, DIV, DIV, DIV, DIV, DIV, DIV, DIV,
  DIV, DIV, DIV, DIV, DIV, DIV, DIV, DIV,

#define _ZETAS_QINV 64
-151046689  , 1830765815  , -1929875198 , -1927777021 , 1640767044  , 1477910808  , 1612161320  , 1640734244  ,
308362795   , -1815525077 , -1374673747 , -1091570561 , -1929495947 , 515185417   , -285697463  , 625853735   ,
1727305304  , 1727305304  , 1727305304  , 1727305304  , 1727305304  , 1727305304  , 1727305304  , 1727305304  ,
2082316400  , 2082316400  , 2082316400  , 2082316400  , 2082316400  , 2082316400  , 2082316400  , 2082316400  ,
-1364982364 , -1364982364 , -1364982364 , -1364982364 , -1364982364 , -1364982364 , -1364982364 , -1364982364 ,
858240904   , 858240904   , 858240904   , 858240904   , 858240904   , 858240904   , 858240904   , 858240904   ,
1806278032  , 1806278032  , 1806278032  , 1806278032  , 1806278032  , 1806278032  , 1806278032  , 1806278032  ,
222489248   , 222489248   , 222489248   , 222489248   , 222489248   , 222489248   , 222489248   , 222489248   ,
-346752664  , -346752664  , -346752664  , -346752664  , -346752664  , -346752664  , -346752664  , -346752664  ,
684667771   , 684667771   , 684667771   , 684667771   , 684667771   , 684667771   , 684667771   , 684667771   ,
1654287830  , 1654287830  , 1654287830  , 1654287830  , 1654287830  , 1654287830  , 1654287830  , 1654287830  ,
-878576921  , -878576921  , -878576921  , -878576921  , -878576921  , -878576921  , -878576921  , -878576921  ,
-1257667337 , -1257667337 , -1257667337 , -1257667337 , -1257667337 , -1257667337 , -1257667337 , -1257667337 ,
-748618600  , -748618600  , -748618600  , -748618600  , -748618600  , -748618600  , -748618600  , -748618600  ,
329347125   , 329347125   , 329347125   , 329347125   , 329347125   , 329347125   , 329347125   , 329347125   ,
1837364258  , 1837364258  , 1837364258  , 1837364258  , 1837364258  , 1837364258  , 1837364258  , 1837364258  ,
-1443016191 , -1443016191 , -1443016191 , -1443016191 , -1443016191 , -1443016191 , -1443016191 , -1443016191 ,
-1170414139 , -1170414139 , -1170414139 , -1170414139 , -1170414139 , -1170414139 , -1170414139 , -1170414139 ,
-1846138265 , -1846138265 , -1846138265 , -1846138265 , -1631226336 , -1631226336 , -1631226336 , -1631226336 ,
-1404529459 , -1404529459 , -1404529459 , -1404529459 , 1838055109  , 1838055109  , 1838055109  , 1838055109  ,
1594295555  , 1594295555  , 1594295555  , 1594295555  , -1076973524 , -1076973524 , -1076973524 , -1076973524 ,
-1898723372 , -1898723372 , -1898723372 , -1898723372 , -594436433  , -594436433  , -594436433  , -594436433  ,
-202001019  , -202001019  , -202001019  , -202001019  , -475984260  , -475984260  , -475984260  , -475984260  ,
-561427818  , -561427818  , -561427818  , -561427818  , 1797021249  , 1797021249  , 1797021249  , 1797021249  ,
-1061813248 , -1061813248 , -1061813248 , -1061813248 , 2059733581  , 2059733581  , 2059733581  , 2059733581  ,
-1661512036 , -1661512036 , -1661512036 , -1661512036 , -1104976547 , -1104976547 , -1104976547 , -1104976547 ,
-1750224323 , -1750224323 , -1750224323 , -1750224323 , -901666090  , -901666090  , -901666090  , -901666090  ,
418987550   , 418987550   , 418987550   , 418987550   , 1831915353  , 1831915353  , 1831915353  , 1831915353  ,
-1925356481 , -1925356481 , -1925356481 , -1925356481 , 992097815   , 992097815   , 992097815   , 992097815   ,
879957084   , 879957084   , 879957084   , 879957084   , 2024403852  , 2024403852  , 2024403852  , 2024403852  ,
1484874664  , 1484874664  , 1484874664  , 1484874664  , -1636082790 , -1636082790 , -1636082790 , -1636082790 ,
-285388938  , -285388938  , -285388938  , -285388938  , -1983539117 , -1983539117 , -1983539117 , -1983539117 ,
-1495136972 , -1495136972 , -1495136972 , -1495136972 , -950076368  , -950076368  , -950076368  , -950076368  ,
-1714807468 , -1714807468 , -1714807468 , -1714807468 , -952438995  , -952438995  , -952438995  , -952438995  ,
-1574918427 , -1574918427 , -654783359  , -654783359  , 1350681039  , 1350681039  , -1974159335 , -1974159335 ,
-2143979939 , -2143979939 , 1651689966  , 1651689966  , 1599739335  , 1599739335  , 140455867   , 140455867   ,
-1285853323 , -1285853323 , -1039411342 , -1039411342 , -993005454  , -993005454  , 1955560694  , 1955560694  ,
-1440787840 , -1440787840 , 1529189038  , 1529189038  , 568627424   , 568627424   , -2131021878 , -2131021878 ,
-783134478  , -783134478  , -247357819  , -247357819  , -588790216  , -588790216  , 1518161567  , 1518161567  ,
289871779   , 289871779   , -86965173   , -86965173   , -1262003603 , -1262003603 , 1708872713  , 1708872713  ,
2135294594  , 2135294594  , 1787797779  , 1787797779  , -1018755525 , -1018755525 , 1638590967  , 1638590967  ,
-889861155  , -889861155  , -120646188  , -120646188  , 1665705315  , 1665705315  , -1669960606 , -1669960606 ,
1321868265  , 1321868265  , -916321552  , -916321552  , 1225434135  , 1225434135  , 1155548552  , 1155548552  ,
-1784632064 , -1784632064 , 2143745726  , 2143745726  , 666258756   , 666258756   , 1210558298  , 1210558298  ,
675310538   , 675310538   , -1261461890 , -1261461890 , -1555941048 , -1555941048 , -318346816  , -318346816  ,
-1999506068 , -1999506068 , 628664287   , 628664287   , -1499481951 , -1499481951 , -1729304568 , -1729304568 ,
-695180180  , -695180180  , 1422575624  , 1422575624  , -1375177022 , -1375177022 , 1424130038  , 1424130038  ,
1777179795  , 1777179795  , -1185330464 , -1185330464 , 334803717   , 334803717   , 235321234   , 235321234   ,
-178766299  , -178766299  , 168022240   , 168022240   , -518252220  , -518252220  , 1206536194  , 1206536194  ,
1957047970  , 1957047970  , 985155484   , 985155484   , 1146323031  , 1146323031  , -894060583  , -894060583  ,
-898413     , 991903578   , 1363007700  , 746144248   , -1363460238 , 912367099   , 30313375    , -1420958686 ,
-605900043  , -44694137   , -326425360  , 2032221021  , 2027833504  , 1176904444  , 1683520342  , 1904936414  ,
14253662    , -421552614  , -517299994  , 1257750362  , 1014493059  , -818371958  , 2027935492  , 1926727420  ,
863641633   , 1747917558  , -1372618620 , 1931587462  , 1819892093  , -325927722  , 128353682   , 1258381762  ,
2124962073  , 908452108   , -1123881663 , 885133339   , -1223601433 , 1851023419  , 137583815   , 1629985060  ,
-1920467227 , -1176751719 , -635454918  , 1967222129  , -1637785316 , -1354528380 , -642772911  , 6363718     ,
-1536588520 , -72690498   , 45766801    , -1287922800 , 694382729   , -314284737  , 671509323   , 1136965286  ,
235104446   , 985022747   , -2070602178 , 1779436847  , -1045062172 , 963438279   , 419615363   , 1116720494  ,
831969619   , -1078959975 , 1216882040  , 1042326957  , -300448763  , 604552167   , -270590488  , 1405999311  ,
756955444   , -1021949428 , -1276805128 , 713994583   , -260312805  , 608791570   , 371462360   , 940195359   ,
1554794072  , 173440395   , -1357098057 , -1542497137 , 1339088280  , -2126092136 , -384158533  , 2061661095  ,
-2040058690 , -1316619236 , 827959816   , -883155599  , -853476187  , -1039370342 , -596344473  , 1726753853  ,
-2047270596 , 6087993     , 702390549   , -1547952704 , -1723816713 , -110126092  , -279505433  , 394851342   ,
-1591599803 , 565464272   , -260424530  , 283780712   , -440824168  , -1758099917 , -71875110   , 776003547   ,
1119856484  , -1600929361 , -1208667171 , 1123958025  , 1544891539  , 879867909   , -1499603926 , 201262505   ,
155290192   , -1809756372 , 2036925262  , 1934038751  , -973777462  , 400711272   , -540420426  , 374860238   ,
#define _ZETAS 592
4808194  , 25847       , -2608894    , -518909     , 237124      , -777960     , -876248     , 466468      ,
1826347     , 2353451     , -359251     , -2091905    , 3119733     , -2884855    , 3111497     , 2680103     ,
2725464     , 2725464     , 2725464     , 2725464     , 2725464     , 2725464     , 2725464     , 2725464     ,
1024112     , 1024112     , 1024112     , 1024112     , 1024112     , 1024112     , 1024112     , 1024112     ,
-1079900    , -1079900    , -1079900    , -1079900    , -1079900    , -1079900    , -1079900    , -1079900    ,
3585928     , 3585928     , 3585928     , 3585928     , 3585928     , 3585928     , 3585928     , 3585928     ,
-549488     , -549488     , -549488     , -549488     , -549488     , -549488     , -549488     , -549488     ,
-1119584    , -1119584    , -1119584    , -1119584    , -1119584    , -1119584    , -1119584    , -1119584    ,
2619752     , 2619752     , 2619752     , 2619752     , 2619752     , 2619752     , 2619752     , 2619752     ,
-2108549    , -2108549    , -2108549    , -2108549    , -2108549    , -2108549    , -2108549    , -2108549    ,
-2118186    , -2118186    , -2118186    , -2118186    , -2118186    , -2118186    , -2118186    , -2118186    ,
-3859737    , -3859737    , -3859737    , -3859737    , -3859737    , -3859737    , -3859737    , -3859737    ,
-1399561    , -1399561    , -1399561    , -1399561    , -1399561    , -1399561    , -1399561    , -1399561    ,
-3277672    , -3277672    , -3277672    , -3277672    , -3277672    , -3277672    , -3277672    , -3277672    ,
1757237     , 1757237     , 1757237     , 1757237     , 1757237     , 1757237     , 1757237     , 1757237     ,
-19422      , -19422      , -19422      , -19422      , -19422      , -19422      , -19422      , -19422      ,
4010497     , 4010497     , 4010497     , 4010497     , 4010497     , 4010497     , 4010497     , 4010497     ,
280005      , 280005      , 280005      , 280005      , 280005      , 280005      , 280005      , 280005      ,
2706023     , 2706023     , 2706023     , 2706023     , 95776       , 95776       , 95776       , 95776       ,
3077325     , 3077325     , 3077325     , 3077325     , 3530437     , 3530437     , 3530437     , 3530437     ,
-1661693    , -1661693    , -1661693    , -1661693    , -3592148    , -3592148    , -3592148    , -3592148    ,
-2537516    , -2537516    , -2537516    , -2537516    , 3915439     , 3915439     , 3915439     , 3915439     ,
-3861115    , -3861115    , -3861115    , -3861115    , -3043716    , -3043716    , -3043716    , -3043716    ,
3574422     , 3574422     , 3574422     , 3574422     , -2867647    , -2867647    , -2867647    , -2867647    ,
3539968     , 3539968     , 3539968     , 3539968     , -300467     , -300467     , -300467     , -300467     ,
2348700     , 2348700     , 2348700     , 2348700     , -539299     , -539299     , -539299     , -539299     ,
-1699267    , -1699267    , -1699267    , -1699267    , -1643818    , -1643818    , -1643818    , -1643818    ,
3505694     , 3505694     , 3505694     , 3505694     , -3821735    , -3821735    , -3821735    , -3821735    ,
3507263     , 3507263     , 3507263     , 3507263     , -2140649    , -2140649    , -2140649    , -2140649    ,
-1600420    , -1600420    , -1600420    , -1600420    , 3699596     , 3699596     , 3699596     , 3699596     ,
811944      , 811944      , 811944      , 811944      , 531354      , 531354      , 531354      , 531354      ,
954230      , 954230      , 954230      , 954230      , 3881043     , 3881043     , 3881043     , 3881043     ,
3900724     , 3900724     , 3900724     , 3900724     , -2556880    , -2556880    , -2556880    , -2556880    ,
2071892     , 2071892     , 2071892     , 2071892     , -2797779    , -2797779    , -2797779    , -2797779    ,
-3930395    , -3930395    , -1528703    , -1528703    , -3677745    , -3677745    , -3041255    , -3041255    ,
-1452451    , -1452451    , 3475950     , 3475950     , 2176455     , 2176455     , -1585221    , -1585221    ,
-1257611    , -1257611    , 1939314     , 1939314     , -4083598    , -4083598    , -1000202    , -1000202    ,
-3190144    , -3190144    , -3157330    , -3157330    , -3632928    , -3632928    , 126922      , 126922      ,
3412210     , 3412210     , -983419     , -983419     , 2147896     , 2147896     , 2715295     , 2715295     ,
-2967645    , -2967645    , -3693493    , -3693493    , -411027     , -411027     , -2477047    , -2477047    ,
-671102     , -671102     , -1228525    , -1228525    , -22981      , -22981      , -1308169    , -1308169    ,
-381987     , -381987     , 1349076     , 1349076     , 1852771     , 1852771     , -1430430    , -1430430    ,
-3343383    , -3343383    , 264944      , 264944      , 508951      , 508951      , 3097992     , 3097992     ,
44288       , 44288       , -1100098    , -1100098    , 904516      , 904516      , 3958618     , 3958618     ,
-3724342    , -3724342    , -8578       , -8578       , 1653064     , 1653064     , -3249728    , -3249728    ,
2389356     , 2389356     , -210977     , -210977     , 759969      , 759969      , -1316856    , -1316856    ,
189548      , 189548      , -3553272    , -3553272    , 3159746     , 3159746     , -1851402    , -1851402    ,
-2409325    , -2409325    , -177440     , -177440     , 1315589     , 1315589     , 1341330     , 1341330     ,
1285669     , 1285669     , -1584928    , -1584928    , -812732     , -812732     , -1439742    , -1439742    ,
-3019102    , -3019102    , -3881060    , -3881060    , -3628969    , -3628969    , 3839961     , 3839961     ,
2091667     , 3407706     , 2316500     , 3817976     , -3342478    , 2244091     , -2446433    , -3562462    ,
266997      , 2434439     , -1235728    , 3513181     , -3520352    , -3759364    , -1197226    , -3193378    ,
900702      , 1859098     , 909542      , 819034      , 495491      , -1613174    , -43260      , -522500     ,
-655327     , -3122442    , 2031748     , 3207046     , -3556995    , -525098     , -768622     , -3595838    ,
342297      , 286988      , -2437823    , 4108315     , 3437287     , -3342277    , 1735879     , 203044      ,
2842341     , 2691481     , -2590150    , 1265009     , 4055324     , 1247620     , 2486353     , 1595974     ,
-3767016    , 1250494     , 2635921     , -3548272    , -2994039    , 1869119     , 1903435     , -1050970    ,
-1333058    , 1237275     , -3318210    , -1430225    , -451100     , 1312455     , 3306115     , -1962642    ,
-1279661    , 1917081     , -2546312    , -1374803    , 1500165     , 777191      , 2235880     , 3406031     ,
-542412     , -2831860    , -1671176    , -1846953    , -2584293    , -3724270    , 594136      , -3776993    ,
-2013608    , 2432395     , 2454455     , -164721     , 1957272     , 3369112     , 185531      , -1207385    ,
-3183426    , 162844      , 1616392     , 3014001     , 810149      , 1652634     , -3694233    , -1799107    ,
-3038916    , 3523897     , 3866901     , 269760      , 2213111     , -975884     , 1717735     , 472078      ,
-426683     , 1723600     , -1803090    , 1910376     , -1667432    , -1104333    , -260646     , -3833893    ,
-2939036    , -2235985    , -420899     , -2286327    , 183443      , -976891     , 1612842     , -3545687    ,
-554416     , 3919660     , -48306      , -1362209    , 3937738     , 1400424     , -846154     , 1976782     ,
 #define _ZETASINV_QINV 1120
 -374860238  , 540420426   , -400711272  , 973777462   , -1934038751 , -2036925262 , 1809756372  , -155290192  ,
-201262505  , 1499603926  , -879867909  , -1544891539 , -1123958025 , 1208667171  , 1600929361  , -1119856484 ,
-776003547  , 71875110    , 1758099917  , 440824168   , -283780712  , 260424530   , -565464272  , 1591599803  ,
-394851342  , 279505433   , 110126092   , 1723816713  , 1547952704  , -702390549  , -6087993    , 2047270596  ,
-1726753853 , 596344473   , 1039370342  , 853476187   , 883155599   , -827959816  , 1316619236  , 2040058690  ,
-2061661095 , 384158533   , 2126092136  , -1339088280 , 1542497137  , 1357098057  , -173440395  , -1554794072 ,
-940195359  , -371462360  , -608791570  , 260312805   , -713994583  , 1276805128  , 1021949428  , -756955444  ,
-1405999311 , 270590488   , -604552167  , 300448763   , -1042326957 , -1216882040 , 1078959975  , -831969619  ,
-1116720494 , -419615363  , -963438279  , 1045062172  , -1779436847 , 2070602178  , -985022747  , -235104446  ,
-1136965286 , -671509323  , 314284737   , -694382729  , 1287922800  , -45766801   , 72690498    , 1536588520  ,
-6363718    , 642772911   , 1354528380  , 1637785316  , -1967222129 , 635454918   , 1176751719  , 1920467227  ,
-1629985060 , -137583815  , -1851023419 , 1223601433  , -885133339  , 1123881663  , -908452108  , -2124962073 ,
-1258381762 , -128353682  , 325927722   , -1819892093 , -1931587462 , 1372618620  , -1747917558 , -863641633  ,
-1926727420 , -2027935492 , 818371958   , -1014493059 , -1257750362 , 517299994   , 421552614   , -14253662   ,
-1904936414 , -1683520342 , -1176904444 , -2027833504 , -2032221021 , 326425360   , 44694137    , 605900043   ,
1420958686  , -30313375   , -912367099  , 1363460238  , -746144248  , -1363007700 , -991903578  , 898413      ,
894060583   , 894060583   , -1146323031 , -1146323031 , -985155484  , -985155484  , -1957047970 , -1957047970 ,
-1206536194 , -1206536194 , 518252220   , 518252220   , -168022240  , -168022240  , 178766299   , 178766299   ,
-235321234  , -235321234  , -334803717  , -334803717  , 1185330464  , 1185330464  , -1777179795 , -1777179795 ,
-1424130038 , -1424130038 , 1375177022  , 1375177022  , -1422575624 , -1422575624 , 695180180   , 695180180   ,
1729304568  , 1729304568  , 1499481951  , 1499481951  , -628664287  , -628664287  , 1999506068  , 1999506068  ,
318346816   , 318346816   , 1555941048  , 1555941048  , 1261461890  , 1261461890  , -675310538  , -675310538  ,
-1210558298 , -1210558298 , -666258756  , -666258756  , -2143745726 , -2143745726 , 1784632064  , 1784632064  ,
-1155548552 , -1155548552 , -1225434135 , -1225434135 , 916321552   , 916321552   , -1321868265 , -1321868265 ,
1669960606  , 1669960606  , -1665705315 , -1665705315 , 120646188   , 120646188   , 889861155   , 889861155   ,
-1638590967 , -1638590967 , 1018755525  , 1018755525  , -1787797779 , -1787797779 , -2135294594 , -2135294594 ,
-1708872713 , -1708872713 , 1262003603  , 1262003603  , 86965173    , 86965173    , -289871779  , -289871779  ,
-1518161567 , -1518161567 , 588790216   , 588790216   , 247357819   , 247357819   , 783134478   , 783134478   ,
2131021878  , 2131021878  , -568627424  , -568627424  , -1529189038 , -1529189038 , 1440787840  , 1440787840  ,
-1955560694 , -1955560694 , 993005454   , 993005454   , 1039411342  , 1039411342  , 1285853323  , 1285853323  ,
-140455867  , -140455867  , -1599739335 , -1599739335 , -1651689966 , -1651689966 , 2143979939  , 2143979939  ,
1974159335  , 1974159335  , -1350681039 , -1350681039 , 654783359   , 654783359   , 1574918427  , 1574918427  ,
952438995   , 952438995   , 952438995   , 952438995   , 1714807468  , 1714807468  , 1714807468  , 1714807468  ,
950076368   , 950076368   , 950076368   , 950076368   , 1495136972  , 1495136972  , 1495136972  , 1495136972  ,
1983539117  , 1983539117  , 1983539117  , 1983539117  , 285388938   , 285388938   , 285388938   , 285388938   ,
1636082790  , 1636082790  , 1636082790  , 1636082790  , -1484874664 , -1484874664 , -1484874664 , -1484874664 ,
-2024403852 , -2024403852 , -2024403852 , -2024403852 , -879957084  , -879957084  , -879957084  , -879957084  ,
-992097815  , -992097815  , -992097815  , -992097815  , 1925356481  , 1925356481  , 1925356481  , 1925356481  ,
-1831915353 , -1831915353 , -1831915353 , -1831915353 , -418987550  , -418987550  , -418987550  , -418987550  ,
901666090   , 901666090   , 901666090   , 901666090   , 1750224323  , 1750224323  , 1750224323  , 1750224323  ,
1104976547  , 1104976547  , 1104976547  , 1104976547  , 1661512036  , 1661512036  , 1661512036  , 1661512036  ,
-2059733581 , -2059733581 , -2059733581 , -2059733581 , 1061813248  , 1061813248  , 1061813248  , 1061813248  ,
-1797021249 , -1797021249 , -1797021249 , -1797021249 , 561427818   , 561427818   , 561427818   , 561427818   ,
475984260   , 475984260   , 475984260   , 475984260   , 202001019   , 202001019   , 202001019   , 202001019   ,
594436433   , 594436433   , 594436433   , 594436433   , 1898723372  , 1898723372  , 1898723372  , 1898723372  ,
1076973524  , 1076973524  , 1076973524  , 1076973524  , -1594295555 , -1594295555 , -1594295555 , -1594295555 ,
-1838055109 , -1838055109 , -1838055109 , -1838055109 , 1404529459  , 1404529459  , 1404529459  , 1404529459  ,
1631226336  , 1631226336  , 1631226336  , 1631226336  , 1846138265  , 1846138265  , 1846138265  , 1846138265  ,
1170414139  , 1170414139  , 1170414139  , 1170414139  , 1170414139  , 1170414139  , 1170414139  , 1170414139  ,
1443016191  , 1443016191  , 1443016191  , 1443016191  , 1443016191  , 1443016191  , 1443016191  , 1443016191  ,
-1837364258 , -1837364258 , -1837364258 , -1837364258 , -1837364258 , -1837364258 , -1837364258 , -1837364258 ,
-329347125  , -329347125  , -329347125  , -329347125  , -329347125  , -329347125  , -329347125  , -329347125  ,
748618600   , 748618600   , 748618600   , 748618600   , 748618600   , 748618600   , 748618600   , 748618600   ,
1257667337  , 1257667337  , 1257667337  , 1257667337  , 1257667337  , 1257667337  , 1257667337  , 1257667337  ,
878576921   , 878576921   , 878576921   , 878576921   , 878576921   , 878576921   , 878576921   , 878576921   ,
-1654287830 , -1654287830 , -1654287830 , -1654287830 , -1654287830 , -1654287830 , -1654287830 , -1654287830 ,
-684667771  , -684667771  , -684667771  , -684667771  , -684667771  , -684667771  , -684667771  , -684667771  ,
346752664   , 346752664   , 346752664   , 346752664   , 346752664   , 346752664   , 346752664   , 346752664   ,
-222489248  , -222489248  , -222489248  , -222489248  , -222489248  , -222489248  , -222489248  , -222489248  ,
-1806278032 , -1806278032 , -1806278032 , -1806278032 , -1806278032 , -1806278032 , -1806278032 , -1806278032 ,
-858240904  , -858240904  , -858240904  , -858240904  , -858240904  , -858240904  , -858240904  , -858240904  ,
1364982364  , 1364982364  , 1364982364  , 1364982364  , 1364982364  , 1364982364  , 1364982364  , 1364982364  ,
-2082316400 , -2082316400 , -2082316400 , -2082316400 , -2082316400 , -2082316400 , -2082316400 , -2082316400 ,
-1727305304 , -1727305304 , -1727305304 , -1727305304 , -1727305304 , -1727305304 , -1727305304 , -1727305304 ,
-625853735  , 285697463   , -515185417  , 1929495947  , 1091570561  , 1374673747  , 1815525077  , -308362795  ,
-1640734244 , -1612161320 , -1477910808 , -1640767044 , 1927777021  , 1929875198  , 151046689 , -8395782   ,
 #define _ZETASINV 1648
-1976782    , 846154      , -1400424    , -3937738    , 1362209     , 48306       , -3919660    , 554416      ,
3545687     , -1612842    , 976891      , -183443     , 2286327     , 420899      , 2235985     , 2939036     ,
3833893     , 260646      , 1104333     , 1667432     , -1910376    , 1803090     , -1723600    , 426683      ,
-472078     , -1717735    , 975884      , -2213111    , -269760     , -3866901    , -3523897    , 3038916     ,
1799107     , 3694233     , -1652634    , -810149     , -3014001    , -1616392    , -162844     , 3183426     ,
1207385     , -185531     , -3369112    , -1957272    , 164721      , -2454455    , -2432395    , 2013608     ,
3776993     , -594136     , 3724270     , 2584293     , 1846953     , 1671176     , 2831860     , 542412      ,
-3406031    , -2235880    , -777191     , -1500165    , 1374803     , 2546312     , -1917081    , 1279661     ,
1962642     , -3306115    , -1312455    , 451100      , 1430225     , 3318210     , -1237275    , 1333058     ,
1050970     , -1903435    , -1869119    , 2994039     , 3548272     , -2635921    , -1250494    , 3767016     ,
-1595974    , -2486353    , -1247620    , -4055324    , -1265009    , 2590150     , -2691481    , -2842341    ,
-203044     , -1735879    , 3342277     , -3437287    , -4108315    , 2437823     , -286988     , -342297     ,
3595838     , 768622      , 525098      , 3556995     , -3207046    , -2031748    , 3122442     , 655327      ,
522500      , 43260       , 1613174     , -495491     , -819034     , -909542     , -1859098    , -900702     ,
3193378     , 1197226     , 3759364     , 3520352     , -3513181    , 1235728     , -2434439    , -266997     ,
3562462     , 2446433     , -2244091    , 3342478     , -3817976    , -2316500    , -3407706    , -2091667    ,
-3839961    , -3839961    , 3628969     , 3628969     , 3881060     , 3881060     , 3019102     , 3019102     ,
1439742     , 1439742     , 812732      , 812732      , 1584928     , 1584928     , -1285669    , -1285669    ,
-1341330    , -1341330    , -1315589    , -1315589    , 177440      , 177440      , 2409325     , 2409325     ,
1851402     , 1851402     , -3159746    , -3159746    , 3553272     , 3553272     , -189548     , -189548     ,
1316856     , 1316856     , -759969     , -759969     , 210977      , 210977      , -2389356    , -2389356    ,
3249728     , 3249728     , -1653064    , -1653064    , 8578        , 8578        , 3724342     , 3724342     ,
-3958618    , -3958618    , -904516     , -904516     , 1100098     , 1100098     , -44288      , -44288      ,
-3097992    , -3097992    , -508951     , -508951     , -264944     , -264944     , 3343383     , 3343383     ,
1430430     , 1430430     , -1852771    , -1852771    , -1349076    , -1349076    , 381987      , 381987      ,
1308169     , 1308169     , 22981       , 22981       , 1228525     , 1228525     , 671102      , 671102      ,
2477047     , 2477047     , 411027      , 411027      , 3693493     , 3693493     , 2967645     , 2967645     ,
-2715295    , -2715295    , -2147896    , -2147896    , 983419      , 983419      , -3412210    , -3412210    ,
-126922     , -126922     , 3632928     , 3632928     , 3157330     , 3157330     , 3190144     , 3190144     ,
1000202     , 1000202     , 4083598     , 4083598     , -1939314    , -1939314    , 1257611     , 1257611     ,
1585221     , 1585221     , -2176455    , -2176455    , -3475950    , -3475950    , 1452451     , 1452451     ,
3041255     , 3041255     , 3677745     , 3677745     , 1528703     , 1528703     , 3930395     , 3930395     ,
2797779     , 2797779     , 2797779     , 2797779     , -2071892    , -2071892    , -2071892    , -2071892    ,
2556880     , 2556880     , 2556880     , 2556880     , -3900724    , -3900724    , -3900724    , -3900724    ,
-3881043    , -3881043    , -3881043    , -3881043    , -954230     , -954230     , -954230     , -954230     ,
-531354     , -531354     , -531354     , -531354     , -811944     , -811944     , -811944     , -811944     ,
-3699596    , -3699596    , -3699596    , -3699596    , 1600420     , 1600420     , 1600420     , 1600420     ,
2140649     , 2140649     , 2140649     , 2140649     , -3507263    , -3507263    , -3507263    , -3507263    ,
3821735     , 3821735     , 3821735     , 3821735     , -3505694    , -3505694    , -3505694    , -3505694    ,
1643818     , 1643818     , 1643818     , 1643818     , 1699267     , 1699267     , 1699267     , 1699267     ,
539299      , 539299      , 539299      , 539299      , -2348700    , -2348700    , -2348700    , -2348700    ,
300467      , 300467      , 300467      , 300467      , -3539968    , -3539968    , -3539968    , -3539968    ,
2867647     , 2867647     , 2867647     , 2867647     , -3574422    , -3574422    , -3574422    , -3574422    ,
3043716     , 3043716     , 3043716     , 3043716     , 3861115     , 3861115     , 3861115     , 3861115     ,
-3915439    , -3915439    , -3915439    , -3915439    , 2537516     , 2537516     , 2537516     , 2537516     ,
3592148     , 3592148     , 3592148     , 3592148     , 1661693     , 1661693     , 1661693     , 1661693     ,
-3530437    , -3530437    , -3530437    , -3530437    , -3077325    , -3077325    , -3077325    , -3077325    ,
-95776      , -95776      , -95776      , -95776      , -2706023    , -2706023    , -2706023    , -2706023    ,
-280005     , -280005     , -280005     , -280005     , -280005     , -280005     , -280005     , -280005     ,
-4010497    , -4010497    , -4010497    , -4010497    , -4010497    , -4010497    , -4010497    , -4010497    ,
19422       , 19422       , 19422       , 19422       , 19422       , 19422       , 19422       , 19422       ,
-1757237    , -1757237    , -1757237    , -1757237    , -1757237    , -1757237    , -1757237    , -1757237    ,
3277672     , 3277672     , 3277672     , 3277672     , 3277672     , 3277672     , 3277672     , 3277672     ,
1399561     , 1399561     , 1399561     , 1399561     , 1399561     , 1399561     , 1399561     , 1399561     ,
3859737     , 3859737     , 3859737     , 3859737     , 3859737     , 3859737     , 3859737     , 3859737     ,
2118186     , 2118186     , 2118186     , 2118186     , 2118186     , 2118186     , 2118186     , 2118186     ,
2108549     , 2108549     , 2108549     , 2108549     , 2108549     , 2108549     , 2108549     , 2108549     ,
-2619752    , -2619752    , -2619752    , -2619752    , -2619752    , -2619752    , -2619752    , -2619752    ,
1119584     , 1119584     , 1119584     , 1119584     , 1119584     , 1119584     , 1119584     , 1119584     ,
549488      , 549488      , 549488      , 549488      , 549488      , 549488      , 549488      , 549488      ,
-3585928    , -3585928    , -3585928    , -3585928    , -3585928    , -3585928    , -3585928    , -3585928    ,
1079900     , 1079900     , 1079900     , 1079900     , 1079900     , 1079900     , 1079900     , 1079900     ,
-1024112    , -1024112    , -1024112    , -1024112    , -1024112    , -1024112    , -1024112    , -1024112    ,
-2725464    , -2725464    , -2725464    , -2725464    , -2725464    , -2725464    , -2725464    , -2725464    ,
-2680103    , -3111497    , 2884855     , -3119733    , 2091905     , 359251      , -2353451    , -1826347    ,
-466468     , 876248      , 777960      , -237124     , 518909      , 2608894     , 3975713     ,   41978     ,

}};
