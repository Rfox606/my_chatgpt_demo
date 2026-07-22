# Codex浠诲姟涔︼細閮ㄥ垎鍏变韩鍔ㄦ€佸師璇笌瀹為獙鐗瑰紓杩炵画杩涚▼妯″瀷 v3.1锛堝畬鏁村寲涓庢柟娉曚慨姝ｏ級

## 鎬荤洰鏍囥€佹湳璇笌闈炵洰鏍?
v3.1 鍙楠屼笁涓寜椤哄簭鎵ц鐨?Gate銆傚叡浜殑鏄?*涓ユ牸绂荤嚎婧愬疄楠?*璁粌鐨勫洜鏋滃姝ラ娴嬪弬鏁颁笌婧愮墖娈靛姩鎬佸師璇厛楠岋紱鐩爣瀹為獙鍙兘浠庡埌杈剧殑鑷韩绐楀彛涓舰鎴?adapter銆丅OCPD 鐗囨銆佺鏈夌姸鎬佷笌杩炵画杩涚▼銆傛瘡娆℃簮鈫掔洰鏍囪繍琛岀嫭绔嬭繘琛岋紝缁濅笉鎶?Exp1 涓?Exp2 鑱斿悎鍦ㄧ嚎璁粌鎴愬悓涓€涓叡浜富妯″瀷銆?
- **婧愨啋鐩爣**锛氶€夋嫨涓€瀹為獙涓?source锛屼娇鐢ㄥ叾鍓?1,600 涓獥鍙ｈ缁冿紱鐩爣鍙湪鑷繁鐨勬暟鎹埌杈惧悗棰勬祴鍜屾洿鏂般€傛柟鍚戝繀椤诲弻鍚戞姤鍛娿€?- **澶氭棰勬祴**锛氬湪宸插埌杈剧殑 32 涓巻鍙茬獥鍙ｄ笂锛岄娴嬫湭鏉?1銆?銆?6 涓獥鍙ｇ殑涔濅釜鍔涚壒寰佸樊鍒嗐€傞娴嬭宸紝鑰岄潪鍏ㄧ▼鏃堕棿銆丼tage 鎴栫（鎹熼噺锛屾槸 Gate A 鐨勪富鐩爣銆?- **鍔ㄦ€佸師璇?*锛氱敱 BOCPD 纭鐗囨鐨勭墖娈电骇锛堝潎鍊笺€佹枩鐜囥€佸垱鏂版畫宸€佹椿鍔級鎻忚堪绗﹁仛绫讳骇鐢燂紱浠讳綍鍗曠獥鍙?KMeans 鍧囦笉绉颁负鍔ㄦ€佸師璇€?- **绉佹湁鐘舵€?*锛氱洰鏍囧疄楠屽彧鐢ㄥ凡纭鐨勮嚜韬墖娈垫寜 BIC 閫夋嫨 K銆佷腑蹇冦€佽涔夊拰璺緞锛涘畠涓嶄細鎺ユ敹 source state centre/variance/transition锛屼篃涓嶄細瀵归綈 state-ID銆?- **杩炵画杩涚▼**锛氱嫭绔嬩簬 state-ID 鐨勭疮绉洜鏋滆瘉鎹紝鏄惧紡杈撳嚭 cumulative progression銆乤ctivity銆乮nitial prior銆乽ncertainty 涓?delayed-entry 鏀舵暃锛涘畠涓嶆槸婊氬姩 z 鍊笺€佸浐瀹氬垎绫绘垨缁濆纾ㄦ崯閲忋€?
鏈换鍔′笉鎭㈠浜旈樁娈靛垎绫汇€佸叏绋嬫椂闂?rank銆佺浉瀵瑰畬鏁磋繘搴︺€佺粷瀵圭（鎹熸瘮杈冩垨鍙粰鍙樺寲鐐硅€屼笉缁欒繛缁繘绋嬨€?
## 缁濆绂佹椤?
1. Exp1 涓?Exp2 鑱斿悎鍦ㄧ嚎璁粌鍚屼竴涓叡浜富妯″瀷锛?2. 璇诲彇鐩爣鏈€缁堥暱搴︺€佺浉瀵瑰畬鏁磋繘搴︽垨浠讳綍鏈潵鐩爣绐楀彛锛?3. 鐢ㄥ湪绾?SVD 浜х敓浼氭棆杞殑 `shared_z`锛?4. 灏嗗崟绐楀彛 KMeans 绉颁负鍔ㄦ€佸師璇紱
5. 灏嗘粴鍔?z 鍊肩О涓鸿繛缁繘绋嬶紱
6. 鍥哄畾鐩爣鐘舵€佹暟銆佸榻?state-ID 鎴栧鍒?source state centre锛?7. 鐪佺暐 adapter銆丅OCPD銆乸rivate state銆乨elayed entry 鎴?uncertainty锛?8. 浣跨敤 Stage銆佸舰璨屻€佺（灞戙€佺粷瀵圭（鎹熼噺鎴栨湭鏉ョ洰鏍囨暟鎹€夊弬锛?9. 鍦ㄨ繍琛屽悗淇敼涓嬭堪闃堝€硷紱
10. 浠ョ畝鍖栨垨鍗犱綅瀹炵幇浠ｆ浛鍚?Gate 鐨勫畬鏁磋緭鍑恒€?
## 杩愯鍓嶅喕缁撶殑閰嶇疆

杈撳叆鍥哄畾涓?`outputs_continuous_state_v45/results/window_feature_raw_v45.csv` 鐨勪節涓?`F_core_v45` 鍔涚壒寰併€傞殢鏈虹瀛?20260722銆備笁姝?horizon=`[1,4,16]`銆佸巻鍙?32銆乻ource train windows=1600銆乺idge=0.25銆乤dapter learning rate=0.06銆乤dapter warmup=128銆乤dapter delayed-label 鏇存柊锛堟瘡涓€ horizon 鐨勭湡鍊煎埌杈惧悗鎵嶆洿鏂帮級銆佽礋杩佺Щ闃堝€?鐩稿 scratch 璇樊楂?3%銆佽繛缁‘璁?3 娆°€?
BOCPD 浣跨敤 Student-t predictive銆乭azard=1/160銆佹渶澶?run length=256銆佺‘璁?posterior=0.65銆佺‘璁よ繛缁?3銆佹渶灏忕墖娈?16 windows銆傜墖娈靛師璇拰绉佹湁鐘舵€佺殑 BIC candidates 閮芥槸 `[2,3,4,5,6]`锛屾渶灏忕兢鍗犳瘮=3%锛岀洰鏍囩鏈夌姸鎬佸浐瀹氱敤鍏?*鍓?6 涓凡纭鐗囨**鏍″噯锛堜笉瓒虫椂鏄庣‘ FAIL锛屼笉鐢ㄦ湭鏉ヨˉ榻愶級銆?
杩炵画杩涚▼鍒濆 prior 鍙?source 鍓?400 涓彲棰勬祴鍒涙柊鐨勫浐瀹氬垎浣嶅昂搴︼紱increment=`0.65*log1p(adapter innovation energy)+0.35*log1p(activity energy)`锛屽彧绱Н闈炶礋鐨勫綋鏈熷洜鏋?increment锛泆ncertainty 鐢卞姝ユ畫宸鏁ｅ害銆乤dapter support deficit銆丅OCPD run-length entropy 鏋勬垚銆俤elayed entry 鍥哄畾涓?Exp1 cycles=`[0,8000,16000,24000]`銆丒xp2 cycles=`[0,3000,6000,9000]`锛屾瘮杈?latest-entry 鍚庣殑鍥哄畾 200 涓叡鍚屽埌杈剧獥鍙ｏ紝涓嶈鍙栨渶缁堥暱搴︺€?
## 棰勬敞鍐?Gate 鎺ュ彈鏍囧噯

姣忎釜鏂瑰悜鐙珛缁欏嚭 Gate A/B/C 鐨?PASS/FAIL 鍜屽師鍥狅紱鎬?PASS 瑕佹眰鍙屽悜姣忎竴 Gate 閮介€氳繃銆?
| Gate | 閫氳繃鏉′欢 |
|---|---|
| A锛氶潪瀵圭О杩佺Щ | 涓変釜 horizon 鍏ㄩ儴鏈夐潪闆跺彲璇勫垎瑕嗙洊锛汼ource+Adapter 鐩稿 Source Frozen 鐨勫钩鍧?MAE 鏀瑰杽鑷冲皯 1%锛涜嫢 adapter 鐩稿 Target From Scratch 杩炵画涓夋楂?3%锛岃礋杩佺Щ gate 蹇呴』鍚敤锛屼笖鍚敤鍚?adapter 鐨勫钩鍧囪宸笉楂樹簬 scratch 鐨?1.05 鍊嶃€?|
| B锛欱OCPD/鍘熻/绉佹湁鐘舵€?| source 鍜?target 鍚勮嚦灏?3 涓‘璁ょ墖娈碉紱source 鐗囨鍘熻鏈夋晥鏁拌嚦灏?2锛涚洰鏍?K 鐢卞叾鍓?6 涓‘璁ょ墖娈电嫭绔?BIC 閫夋嫨锛?--6锛夛紝鎵€鏈変腑蹇?provenance=`target_confirmed_segments_only`锛屾棤璺ㄥ疄楠?ID 鏄犲皠锛涘師璇緭鍏ヨ鏁板繀椤荤瓑浜庣‘璁ょ墖娈垫暟鑰岄潪绐楀彛鏁般€?|
| C锛氳繛缁繘绋?涓嶇‘瀹氭€?寤惰繜鎺ュ叆 | 姣忎釜鐩爣绐楀彛閮芥湁鏈夐檺 cumulative progression銆乤ctivity銆乮nitial prior銆乽ncertainty锛沘ctivity 鏍囧噯宸?>`1e-6`锛涘悎鎴?OOD 鐨?uncertainty 楂樹簬姝ｅ父杞ㄨ抗鑷冲皯 20%锛涙墍鏈夐娉ㄥ唽 delayed entries 鏈?200 涓叡鍚屽埌杈剧獥鍙ｏ紝latest-entry 鍚庣殑 pairwise increment NRMSE <=0.50锛涜繛缁ā鍧楄鍏?state-ID 娆℃暟=0銆?|
| 閫氱敤 | 涓や釜棰勬敞鍐?prefix cutoff=`[0.35,0.60]` 鐨勯娴嬨€丅OCPD銆佺鏈夌姸鎬佷笌杩炵画杈撳嚭鏈€澶у樊 `<=1e-12`锛汼tage/褰㈣矊/纾ㄥ睉/缁濆纾ㄦ崯/鏈潵鐩爣/鐩爣鏈€缁堥暱搴﹁鍙栨鏁?0锛涘叏閮?pytest 閫氳繃銆?|

姣忔姝ｅ紡杩愯鍚庝笉寰椾负鎻愰珮缁撴灉閲嶈皟鍙傛暟銆備换浣?Gate 澶辫触銆佽礋杩佺Щ瑙﹀彂銆佹牎鍑嗕笉瓒虫垨宸ョ▼澶辫触閮藉繀椤诲畬鏁翠繚鐣欏埌缁撴灉鐩綍鍜岃嚜鍔ㄦ姤鍛娿€?

