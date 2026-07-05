"""
SmartMatch AI — 알고리즘 기반 HR 최적화 대시보드
실행 방법: streamlit run smartmatch_hr.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import numpy as np

# ── 한글 폰트 설정 ──────────────────────────────────────────────
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'NanumGothic', 'AppleGothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 페이지 설정 ─────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartMatch AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 커스텀 CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:28px; font-weight:600; margin-bottom:4px; }
    .sub-title   { color:#888; font-size:14px; margin-bottom:1.5rem; }
    .metric-card { background:#F8F8FF; border-radius:12px; padding:1rem 1.25rem;
                   border:1px solid #E0DCFF; text-align:center; }
    .metric-num  { font-size:28px; font-weight:600; color:#534AB7; }
    .metric-lbl  { font-size:12px; color:#888; margin-top:2px; }
    .emp-card    { background:white; border:1px solid #eee; border-radius:10px;
                   padding:.85rem 1rem; margin-bottom:.5rem; }
    .badge       { display:inline-block; padding:2px 10px; border-radius:12px;
                   font-size:12px; font-weight:500; }
    .nudge-box   { background:#F3F0FF; border-left:4px solid #7F77DD;
                   border-radius:8px; padding:.9rem 1.1rem; margin-top:.75rem; }
    .warn-box    { background:#FFF8EC; border-left:4px solid #EF9F27;
                   border-radius:8px; padding:.75rem 1rem; margin:.5rem 0; }
    .ok-box      { background:#EAF3DE; border-left:4px solid #639922;
                   border-radius:8px; padding:.75rem 1rem; margin:.5rem 0; }
    stButton>button { border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ── 색상 & 상수 ─────────────────────────────────────────────────
SKILL_COLORS = {
    "마케팅/기획": "#B5D4F4",
    "개발/기술":   "#C0DD97",
    "디자인":      "#FAC775",
    "재무/회계":   "#CECBF6",
}
SKILL_TEXT = {
    "마케팅/기획": "#0C447C",
    "개발/기술":   "#27500A",
    "디자인":      "#633806",
    "재무/회계":   "#3C3489",
}
# 역량별 요일 근무 패턴 (월~금)
SKILL_PATTERN = {
    "마케팅/기획": ["오전", "오전", "오후", "오전", "오전"],
    "개발/기술":   ["풀타임", "오전", "풀타임", "오후", "오전"],
    "디자인":      ["오전", "풀타임", "오전", "오전", "오후"],
    "재무/회계":   ["오후", "오전", "오전", "풀타임", "오전"],
}
DAYS = ["월", "화", "수", "목", "금"]
FATIGUE_EMOJI = {1:"🟢", 2:"🟢", 3:"🟡", 4:"🔴", 5:"🔴"}

# ── 세션 상태 초기화 ────────────────────────────────────────────
if "employees" not in st.session_state:
    st.session_state.employees = [
        {"id":1, "name":"김민준", "skill":"개발/기술",   "fatigue":2, "pref":"풀타임"},
        {"id":2, "name":"박지우", "skill":"마케팅/기획", "fatigue":3, "pref":"오전"},
        {"id":3, "name":"이서연", "skill":"디자인",       "fatigue":4, "pref":"오전"},
        {"id":4, "name":"최현우", "skill":"재무/회계",   "fatigue":1, "pref":"오후"},
        {"id":5, "name":"정다은", "skill":"개발/기술",   "fatigue":2, "pref":"풀타임"},
    ]
if "next_id" not in st.session_state:
    st.session_state.next_id = 6
if "del_id"  not in st.session_state:
    st.session_state.del_id = None

# ── 헬퍼 함수 ───────────────────────────────────────────────────
def assign_shift(emp: dict, day_idx: int) -> str:
    """피로도·선호 근무·역량 패턴을 고려해 근무 배치를 결정"""
    if emp["fatigue"] >= 4:
        return ["오전", "휴무", "오전", "휴무", "오전"][day_idx]
    pattern_shift = SKILL_PATTERN[emp["skill"]][day_idx]
    if emp["pref"] != "풀타임" and pattern_shift == "풀타임":
        return emp["pref"]
    return pattern_shift

def calc_optimization_score(emps: list) -> int:
    high = sum(1 for e in emps if e["fatigue"] >= 4)
    avg_f = sum(e["fatigue"] for e in emps) / len(emps)
    score = max(55, round(95 - high * 8 - (avg_f - 2) * 5))
    return score

def get_status(fatigue: int) -> str:
    if fatigue <= 2: return "✅ 정상"
    if fatigue <= 3: return "⚠️ 주의"
    return "🚨 휴식 필요"

def get_nudge(emps: list) -> str:
    high_f = [e for e in emps if e["fatigue"] >= 4]
    avg_f  = sum(e["fatigue"] for e in emps) / len(emps)
    score  = calc_optimization_score(emps)
    if score >= 90:
        return "현재 팀 구성은 매우 이상적입니다. 적재적소 원칙이 잘 지켜지고 있으며 피로도 분포도 균형적입니다."
    if len(high_f) >= 2:
        names = ", ".join(e["name"] for e in high_f)
        return (f"{names} 등 다수 직원의 피로도가 높습니다. "
                "업무 재배분(Job Rotation) 또는 추가 인력 채용을 검토하세요. "
                "장기적으로 번아웃은 거래비용(Transaction Cost)을 상승시킵니다.")
    if avg_f > 3:
        return ("팀 전반의 피로도가 높아지고 있습니다. "
                "유연근무제나 재충전 프로그램 도입을 검토하세요. "
                "인적자원(HR)은 교체 비용이 가장 높은 자산입니다.")
    return ("일부 직원의 피로도가 누적되고 있습니다. "
            "1:1 면담을 통해 업무 부담을 점검하고 선호 업무 배치 비율을 높이세요.")

# ── 사이드바: 직원 추가 ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💼 SmartMatch AI")
    st.markdown("알고리즘 기반 HR 최적화")
    st.divider()

    st.markdown("### ➕ 직원 추가")
    new_name   = st.text_input("이름", placeholder="예: 홍길동")
    new_skill  = st.selectbox("역량", list(SKILL_COLORS.keys()))
    new_pref   = st.selectbox("선호 근무", ["오전", "오후", "풀타임"])
    new_fat    = st.slider("피로도", 1, 5, 2,
                           help="1=매우 좋음 / 3=보통 / 5=번아웃 위험")
    fat_labels = {1:"매우 좋음 🟢", 2:"양호 🟢", 3:"보통 🟡", 4:"피로 누적 🔴", 5:"번아웃 위험 🔴"}
    st.caption(f"현재 피로도: **{fat_labels[new_fat]}**")

    if st.button("등록하기", use_container_width=True, type="primary"):
        if new_name.strip():
            st.session_state.employees.append({
                "id":      st.session_state.next_id,
                "name":    new_name.strip(),
                "skill":   new_skill,
                "fatigue": new_fat,
                "pref":    new_pref,
            })
            st.session_state.next_id += 1
            st.success(f"✅ {new_name} 등록 완료!")
            st.rerun()
        else:
            st.error("이름을 입력해주세요.")

    st.divider()
    st.caption("🎓 고등학생 알고리즘 경영 프로젝트\nSmartMatch AI v1.0")

# ── 메인 헤더 ───────────────────────────────────────────────────
emps = st.session_state.employees
st.markdown('<p class="main-title">💼 SmartMatch AI 경영 시스템</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">알고리즘 기반 직원 직무 매칭 & 근무 스케줄 자동화 대시보드</p>',
            unsafe_allow_html=True)

# ── 지표 카드 ───────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
avg_fat = sum(e["fatigue"] for e in emps) / len(emps) if emps else 0
score   = calc_optimization_score(emps) if emps else 0
high_cnt = sum(1 for e in emps if e["fatigue"] >= 4)

c1.metric("👥 총 직원 수",    f"{len(emps)}명")
c2.metric("😴 평균 피로도",   f"{avg_fat:.1f}점")
c3.metric("🏆 최적화 점수",   f"{score}점")
c4.metric("🚨 주의 인원",     f"{high_cnt}명")

st.divider()

# ── 탭 구성 ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["👥 직원 관리", "📅 스케줄 배치", "📊 AI 분석 리포트"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — 직원 관리
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### 등록된 직원 목록")

    if not emps:
        st.info("등록된 직원이 없습니다. 왼쪽 사이드바에서 직원을 추가해주세요.")
    else:
        # 헤더 행
        hc = st.columns([2, 2.2, 1.2, 1.4, 1.5, 1])
        for col, label in zip(hc, ["이름", "역량", "피로도", "선호 근무", "상태", "삭제"]):
            col.markdown(f"**{label}**")
        st.markdown("---")

        for emp in emps:
            cols = st.columns([2, 2.2, 1.2, 1.4, 1.5, 1])
            cols[0].markdown(f"**{emp['name']}**")

            # 역량 배지 (색상)
            bg  = SKILL_COLORS[emp["skill"]]
            fg  = SKILL_TEXT[emp["skill"]]
            cols[1].markdown(
                f'<span class="badge" style="background:{bg};color:{fg}">'
                f'{emp["skill"]}</span>', unsafe_allow_html=True
            )

            # 피로도 점 시각화
            dots = "●" * emp["fatigue"] + "○" * (5 - emp["fatigue"])
            fat_color = "#639922" if emp["fatigue"] <= 2 else ("#BA7517" if emp["fatigue"] == 3 else "#E24B4A")
            cols[2].markdown(
                f'<span style="color:{fat_color};font-size:13px">{dots}</span> '
                f'<span style="font-size:12px;color:#888">{emp["fatigue"]}점</span>',
                unsafe_allow_html=True
            )

            cols[3].caption(emp["pref"])
            cols[4].caption(get_status(emp["fatigue"]))

            if cols[5].button("🗑", key=f"del_{emp['id']}",
                              help=f"{emp['name']} 삭제"):
                st.session_state.employees = [
                    e for e in st.session_state.employees if e["id"] != emp["id"]
                ]
                st.rerun()

    st.divider()

    # ── 피로도 분포 차트 ──
    st.markdown("#### 직원 피로도 분포")
    if emps:
        fig, ax = plt.subplots(figsize=(8, 2.8))
        names   = [e["name"] for e in emps]
        fatigues = [e["fatigue"] for e in emps]
        bar_colors = [
            "#639922" if f <= 2 else ("#EF9F27" if f == 3 else "#E24B4A")
            for f in fatigues
        ]
        bars = ax.barh(names, fatigues, color=bar_colors, height=0.55)
        ax.set_xlim(0, 5.5)
        ax.set_xlabel("피로도 (1~5점)", fontsize=10)
        ax.axvline(x=3, color="#EF9F27", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.axvline(x=4, color="#E24B4A", linestyle="--", linewidth=1.2, alpha=0.7)
        for bar, val in zip(bars, fatigues):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=10)
        patches = [
            mpatches.Patch(color="#639922", label="정상 (1~2)"),
            mpatches.Patch(color="#EF9F27", label="주의 (3)"),
            mpatches.Patch(color="#E24B4A", label="위험 (4~5)"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=9)
        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("직원을 추가하면 차트가 나타납니다.")

# ═══════════════════════════════════════════════════════════════
# TAB 2 — 스케줄 배치
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 이번 주 최적 근무 스케줄")
    st.caption("알고리즘이 피로도·역량·선호 근무를 고려해 자동 배치합니다.")

    if not emps:
        st.info("직원을 먼저 등록해주세요.")
    else:
        # 피로도 경고
        for emp in emps:
            if emp["fatigue"] >= 4:
                st.markdown(
                    f'<div class="warn-box">⚠️ <b>{emp["name"]}</b>의 피로도가 높아 '
                    f'연속·야간 근무에서 제외했습니다.</div>',
                    unsafe_allow_html=True
                )

        # 스케줄 테이블 생성
        rows = []
        for emp in emps:
            row = {"직원": emp["name"], "역량": emp["skill"]}
            for i, day in enumerate(DAYS):
                row[day] = assign_shift(emp, i)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("직원")

        # 셀 색상 매핑
        SHIFT_COLORS = {
            "오전":  ("background-color:#E6F1FB;color:#185FA5", ),
            "오후":  ("background-color:#EEEDFE;color:#534AB7", ),
            "풀타임":("background-color:#EAF3DE;color:#3B6D11", ),
            "휴무":  ("background-color:#F1EFE8;color:#888780", ),
        }

        def style_shift(val):
            if val in SHIFT_COLORS:
                return SHIFT_COLORS[val][0]
            return ""

        styled = (df.style
                  .applymap(style_shift, subset=DAYS)
                  .set_properties(**{"text-align":"center", "font-size":"13px",
                                     "padding":"6px 12px"})
                  .set_table_styles([
                      {"selector":"th",
                       "props":[("background-color","#F3F0FF"),
                                ("color","#3C3489"),
                                ("font-size","13px"),
                                ("text-align","center"),
                                ("padding","8px 12px")]},
                  ]))
        st.dataframe(styled, use_container_width=True)

        # ── 주별 근무 시간 집계 차트 ──
        st.markdown("#### 직원별 예상 주간 근무 시간")
        SHIFT_HRS = {"오전": 4, "오후": 4, "풀타임": 8, "휴무": 0}
        total_hrs = []
        for emp in emps:
            hrs = sum(SHIFT_HRS[assign_shift(emp, i)] for i in range(5))
            total_hrs.append(hrs)

        fig2, ax2 = plt.subplots(figsize=(8, 2.6))
        bar_c = [SKILL_COLORS[e["skill"]] for e in emps]
        names = [e["name"] for e in emps]
        bars2 = ax2.bar(names, total_hrs, color=bar_c, width=0.5, edgecolor="white")
        ax2.axhline(y=40, color="#E24B4A", linestyle="--", linewidth=1.2,
                    label="법정 근무 한도 (40h)", alpha=0.8)
        ax2.set_ylabel("시간(h)", fontsize=10)
        ax2.set_ylim(0, 48)
        for bar, h in zip(bars2, total_hrs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{h}h", ha="center", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.set_facecolor("#FAFAFA")
        fig2.patch.set_facecolor("white")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        st.pyplot(fig2)
        plt.close(fig2)

        # 범례
        st.markdown("**근무 종류 설명**")
        legend_cols = st.columns(4)
        items = [("🔵 오전","오전 4시간 근무"),("🟣 오후","오후 4시간 근무"),
                 ("🟢 풀타임","8시간 전일 근무"),("⚪ 휴무","해당 일 미배치")]
        for col, (label, desc) in zip(legend_cols, items):
            col.caption(f"{label}: {desc}")

# ═══════════════════════════════════════════════════════════════
# TAB 3 — AI 분석 리포트
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### AI 인사 관리 리포트")

    if not emps:
        st.info("직원을 먼저 등록해주세요.")
    else:
        score = calc_optimization_score(emps)
        high_f = [e for e in emps if e["fatigue"] >= 4]

        # ── 최적화 점수 ──
        score_color = "#639922" if score>=85 else ("#EF9F27" if score>=70 else "#E24B4A")
        col_score, col_info = st.columns([1, 3])
        with col_score:
            st.markdown(
                f'<div style="border:3px solid {score_color};border-radius:50%;'
                f'width:96px;height:96px;display:flex;flex-direction:column;'
                f'align-items:center;justify-content:center;margin:auto">'
                f'<span style="font-size:26px;font-weight:600;color:{score_color}">{score}</span>'
                f'<span style="font-size:11px;color:#888">최적화 점수</span></div>',
                unsafe_allow_html=True
            )
        with col_info:
            st.markdown(f"**팀 최적화 점수: {score}점**")
            if score >= 85:
                st.markdown('<div class="ok-box">✅ 팀 상태가 매우 양호합니다.</div>',
                            unsafe_allow_html=True)
            elif score >= 70:
                st.markdown('<div class="warn-box">⚠️ 일부 직원 피로도 관리가 필요합니다.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="background:#FCEBEB;border-left:4px solid #E24B4A;'
                    'border-radius:8px;padding:.75rem 1rem">🚨 팀 피로도가 위험 수준입니다. '
                    '즉각적인 조치가 필요합니다.</div>',
                    unsafe_allow_html=True
                )

        st.divider()

        # ── 역량별 인력 현황 파이 차트 ──
        st.markdown("#### 역량별 인력 분포")
        from collections import Counter
        skill_cnt = Counter(e["skill"] for e in emps)
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        labels = list(skill_cnt.keys())
        sizes  = list(skill_cnt.values())
        colors = [SKILL_COLORS[s] for s in labels]
        wedges, texts, autotexts = ax3.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.0f%%", startangle=90,
            wedgeprops={"edgecolor":"white","linewidth":2}
        )
        for t in texts:     t.set_fontsize(10)
        for at in autotexts: at.set_fontsize(9)
        fig3.patch.set_facecolor("white")
        st.pyplot(fig3)
        plt.close(fig3)

        # ── 직원별 상세 현황 ──
        st.markdown("#### 직원별 상태 현황")
        detail_data = []
        for emp in emps:
            hrs = sum(
                {"오전":4,"오후":4,"풀타임":8,"휴무":0}[assign_shift(emp, i)]
                for i in range(5)
            )
            detail_data.append({
                "이름":    emp["name"],
                "역량":    emp["skill"],
                "피로도":  f"{emp['fatigue']}점",
                "상태":    get_status(emp["fatigue"]),
                "주간 근무": f"{hrs}h",
                "선호 근무": emp["pref"],
            })
        st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)

        # ── 경영 넛지 조언 ──
        st.divider()
        st.markdown("#### 🤖 경영 넛지 조언")
        nudge = get_nudge(emps)
        st.markdown(f'<div class="nudge-box">💡 {nudge}</div>', unsafe_allow_html=True)

        # ── 피로 위험 직원 상세 경고 ──
        if high_f:
            st.markdown("#### 🚨 주의 대상 직원 상세")
            for emp in high_f:
                with st.expander(f"⚠️ {emp['name']} — 피로도 {emp['fatigue']}점"):
                    st.markdown(f"""
- **현재 역량**: {emp['skill']}  
- **피로도**: {emp['fatigue']}점 / 5점  
- **권장 조치**: {'즉시 휴식 또는 업무 감소 필요' if emp['fatigue']==5 else '연속 근무 제한 및 경과 관찰'}  
- **배치 조정**: 야간/연속 근무 자동 제외, 주 2회 휴무로 배정됨
                    """)

        st.divider()
        st.markdown(
            '<p style="font-size:12px;color:#aaa;text-align:center">'
            '⚠️ 본 시스템은 알고리즘 경영 학습용입니다. '
            '실제 인사 결정에는 관리자의 최종 판단이 필요합니다.</p>',
            unsafe_allow_html=True
        )
