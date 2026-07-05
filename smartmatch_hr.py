"""
SmartMatch AI — 알고리즘 기반 HR 최적화 대시보드
실행 방법: streamlit run smartmatch_hr.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np
import os
import urllib.request
from collections import Counter

# ── 한글 폰트 자동 설치 ────────────────────────────────────────
@st.cache_resource
def load_korean_font():
    font_path = "/tmp/NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        urllib.request.urlretrieve(url, font_path)
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    return prop.get_name()

korean_font = load_korean_font()
matplotlib.rcParams['font.family'] = korean_font
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 페이지 설정 ─────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartMatch AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 커스텀 CSS (다크모드 대응 포함) ────────────────────────────
st.markdown("""
<style>
    /* 다크모드에서도 텍스트 강제 색상 지정 */
    .force-dark {
        color: #1a1a1a !important;
    }
    .main-title {
        font-size:28px; font-weight:600; margin-bottom:4px;
    }
    .sub-title {
        color:#888; font-size:14px; margin-bottom:1.5rem;
    }
    .badge {
        display:inline-block; padding:2px 10px; border-radius:12px;
        font-size:12px; font-weight:600;
    }
    /* 경고/상태 박스 — 텍스트 색상 명시 */
    .nudge-box {
        background:#3a2f6e; border-left:4px solid #9b8fe8;
        border-radius:8px; padding:.9rem 1.1rem; margin-top:.75rem;
        color:#e8e3ff !important;
    }
    .warn-box {
        background:#3d2f00; border-left:4px solid #EF9F27;
        border-radius:8px; padding:.75rem 1rem; margin:.5rem 0;
        color:#ffd98a !important;
    }
    .ok-box {
        background:#1a3010; border-left:4px solid #639922;
        border-radius:8px; padding:.75rem 1rem; margin:.5rem 0;
        color:#b8e88a !important;
    }
    .err-box {
        background:#3d0f0f; border-left:4px solid #E24B4A;
        border-radius:8px; padding:.75rem 1rem; margin:.5rem 0;
        color:#ffaaaa !important;
    }
    /* 스케줄 표 셀 텍스트 강제 */
    .shift-am   { background:#1a3a5c; color:#7ec8f8 !important; padding:3px 10px; border-radius:8px; font-weight:600; display:inline-block; }
    .shift-pm   { background:#2d1a5c; color:#c0b0ff !important; padding:3px 10px; border-radius:8px; font-weight:600; display:inline-block; }
    .shift-full { background:#1a3a10; color:#8fdd50 !important; padding:3px 10px; border-radius:8px; font-weight:600; display:inline-block; }
    .shift-off  { background:#2a2a2a; color:#aaaaaa !important; padding:3px 10px; border-radius:8px; font-weight:600; display:inline-block; }
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
    "개발/기술":   "#1a4a05",
    "디자인":      "#633806",
    "재무/회계":   "#3C3489",
}
SKILL_PATTERN = {
    "마케팅/기획": ["오전", "오전", "오후", "오전", "오전"],
    "개발/기술":   ["풀타임", "오전", "풀타임", "오후", "오전"],
    "디자인":      ["오전", "풀타임", "오전", "오전", "오후"],
    "재무/회계":   ["오후", "오전", "오전", "풀타임", "오전"],
}
DAYS = ["월", "화", "수", "목", "금"]

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

# ── 헬퍼 함수 ───────────────────────────────────────────────────
def assign_shift(emp, day_idx):
    if emp["fatigue"] >= 4:
        return ["오전", "휴무", "오전", "휴무", "오전"][day_idx]
    pattern_shift = SKILL_PATTERN[emp["skill"]][day_idx]
    if emp["pref"] != "풀타임" and pattern_shift == "풀타임":
        return emp["pref"]
    return pattern_shift

def calc_score(emps):
    if not emps: return 0
    high = sum(1 for e in emps if e["fatigue"] >= 4)
    avg_f = sum(e["fatigue"] for e in emps) / len(emps)
    return max(55, round(95 - high * 8 - (avg_f - 2) * 5))

def get_status(f):
    if f <= 2: return "✅ 정상"
    if f <= 3: return "⚠️ 주의"
    return "🚨 휴식 필요"

def get_nudge(emps):
    high_f = [e for e in emps if e["fatigue"] >= 4]
    avg_f  = sum(e["fatigue"] for e in emps) / len(emps)
    score  = calc_score(emps)
    if score >= 90:
        return "현재 팀 구성은 매우 이상적입니다. 적재적소 원칙이 잘 지켜지고 있으며 피로도 분포도 균형적입니다."
    if len(high_f) >= 2:
        names = ", ".join(e["name"] for e in high_f)
        return f"{names} 등 다수 직원의 피로도가 높습니다. 업무 재배분(Job Rotation) 또는 추가 인력 채용을 검토하세요. 장기적으로 번아웃은 거래비용(Transaction Cost)을 상승시킵니다."
    if avg_f > 3:
        return "팀 전반의 피로도가 높아지고 있습니다. 유연근무제나 재충전 프로그램 도입을 검토하세요. 인적자원(HR)은 교체 비용이 가장 높은 자산입니다."
    return "일부 직원의 피로도가 누적되고 있습니다. 1:1 면담을 통해 업무 부담을 점검하고 선호 업무 배치 비율을 높이세요."

def shift_badge(shift):
    cls = {"오전":"shift-am","오후":"shift-pm","풀타임":"shift-full","휴무":"shift-off"}
    c = cls.get(shift, "shift-off")
    return f'<span class="{c}">{shift}</span>'

# ── 차트 공통 배경색 ────────────────────────────────────────────
CHART_BG   = "#1a1a2e"   # 차트 영역
CHART_FG   = "#e0e0e0"   # 축 텍스트
CHART_GRID = "#2e2e4e"   # 그리드

def apply_dark_style(fig, ax):
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=CHART_FG)
    ax.xaxis.label.set_color(CHART_FG)
    ax.yaxis.label.set_color(CHART_FG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CHART_GRID)
    ax.spines["bottom"].set_color(CHART_GRID)

# ── 사이드바 ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💼 SmartMatch AI")
    st.markdown("알고리즘 기반 HR 최적화")
    st.divider()
    st.markdown("### ➕ 직원 추가")
    new_name  = st.text_input("이름", placeholder="예: 홍길동")
    new_skill = st.selectbox("역량", list(SKILL_COLORS.keys()))
    new_pref  = st.selectbox("선호 근무", ["오전", "오후", "풀타임"])
    new_fat   = st.slider("피로도", 1, 5, 2, help="1=매우 좋음 / 3=보통 / 5=번아웃 위험")
    fat_labels = {1:"매우 좋음 🟢",2:"양호 🟢",3:"보통 🟡",4:"피로 누적 🔴",5:"번아웃 위험 🔴"}
    st.caption(f"현재 피로도: **{fat_labels[new_fat]}**")
    if st.button("등록하기", use_container_width=True, type="primary"):
        if new_name.strip():
            st.session_state.employees.append({
                "id": st.session_state.next_id,
                "name": new_name.strip(),
                "skill": new_skill,
                "fatigue": new_fat,
                "pref": new_pref,
            })
            st.session_state.next_id += 1
            st.success(f"✅ {new_name} 등록 완료!")
            st.rerun()
        else:
            st.error("이름을 입력해주세요.")
    st.divider()
    st.caption("🎓 고등학생 알고리즘 경영 프로젝트\nSmartMatch AI v1.1")

# ── 메인 헤더 ───────────────────────────────────────────────────
emps = st.session_state.employees
st.markdown('<p class="main-title">💼 SmartMatch AI 경영 시스템</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">알고리즘 기반 직원 직무 매칭 & 근무 스케줄 자동화 대시보드</p>', unsafe_allow_html=True)

avg_fat  = sum(e["fatigue"] for e in emps) / len(emps) if emps else 0
score    = calc_score(emps)
high_cnt = sum(1 for e in emps if e["fatigue"] >= 4)

c1, c2, c3, c4 = st.columns(4)
c1.metric("👥 총 직원 수",  f"{len(emps)}명")
c2.metric("😴 평균 피로도", f"{avg_fat:.1f}점")
c3.metric("🏆 최적화 점수", f"{score}점")
c4.metric("🚨 주의 인원",   f"{high_cnt}명")
st.divider()

# ── 탭 ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["👥 직원 관리", "📅 스케줄 배치", "📊 AI 분석 리포트"])

# ═══════════════════════════════════════════════
# TAB 1 — 직원 관리
# ═══════════════════════════════════════════════
with tab1:
    st.markdown("#### 등록된 직원 목록")
    if not emps:
        st.info("등록된 직원이 없습니다. 왼쪽 사이드바에서 직원을 추가해주세요.")
    else:
        hc = st.columns([2, 2.2, 1.2, 1.4, 1.5, 1])
        for col, label in zip(hc, ["이름", "역량", "피로도", "선호 근무", "상태", "삭제"]):
            col.markdown(f"**{label}**")
        st.markdown("---")
        for emp in emps:
            cols = st.columns([2, 2.2, 1.2, 1.4, 1.5, 1])
            cols[0].markdown(f"**{emp['name']}**")
            bg = SKILL_COLORS[emp["skill"]]
            fg = SKILL_TEXT[emp["skill"]]
            cols[1].markdown(
                f'<span class="badge" style="background:{bg};color:{fg}">{emp["skill"]}</span>',
                unsafe_allow_html=True
            )
            dots = "●" * emp["fatigue"] + "○" * (5 - emp["fatigue"])
            fat_color = "#7dd94a" if emp["fatigue"] <= 2 else ("#f5c842" if emp["fatigue"] == 3 else "#f55a5a")
            cols[2].markdown(
                f'<span style="color:{fat_color};font-size:14px">{dots}</span> '
                f'<span style="font-size:12px;color:#aaa">{emp["fatigue"]}점</span>',
                unsafe_allow_html=True
            )
            cols[3].caption(emp["pref"])
            cols[4].caption(get_status(emp["fatigue"]))
            if cols[5].button("🗑", key=f"del_{emp['id']}", help=f"{emp['name']} 삭제"):
                st.session_state.employees = [e for e in st.session_state.employees if e["id"] != emp["id"]]
                st.rerun()

    st.divider()
    st.markdown("#### 직원 피로도 분포")
    if emps:
        fig, ax = plt.subplots(figsize=(8, 2.8))
        names    = [e["name"] for e in emps]
        fatigues = [e["fatigue"] for e in emps]
        bar_colors = ["#5db832" if f<=2 else ("#e8a020" if f==3 else "#d94040") for f in fatigues]
        bars = ax.barh(names, fatigues, color=bar_colors, height=0.55)
        ax.set_xlim(0, 5.5)
        ax.set_xlabel("피로도 (1~5점)", fontsize=10, color=CHART_FG)
        ax.axvline(x=3, color="#e8a020", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.axvline(x=4, color="#d94040", linestyle="--", linewidth=1.2, alpha=0.7)
        for bar, val in zip(bars, fatigues):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=10, color=CHART_FG)
        ax.set_yticklabels(names, color=CHART_FG)
        patches = [
            mpatches.Patch(color="#5db832", label="정상 (1~2)"),
            mpatches.Patch(color="#e8a020", label="주의 (3)"),
            mpatches.Patch(color="#d94040", label="위험 (4~5)"),
        ]
        legend = ax.legend(handles=patches, loc="lower right", fontsize=9,
                           facecolor=CHART_BG, edgecolor=CHART_GRID, labelcolor=CHART_FG)
        apply_dark_style(fig, ax)
        st.pyplot(fig)
        plt.close(fig)

# ═══════════════════════════════════════════════
# TAB 2 — 스케줄 배치
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("#### 이번 주 최적 근무 스케줄")
    st.caption("알고리즘이 피로도·역량·선호 근무를 고려해 자동 배치합니다.")
    if not emps:
        st.info("직원을 먼저 등록해주세요.")
    else:
        for emp in emps:
            if emp["fatigue"] >= 4:
                st.markdown(
                    f'<div class="warn-box">⚠️ <b>{emp["name"]}</b>의 피로도가 높아 연속·야간 근무에서 제외했습니다.</div>',
                    unsafe_allow_html=True
                )

        # HTML 테이블로 직접 렌더링 (다크모드 완벽 대응)
        table_html = """
        <table style="width:100%;border-collapse:collapse;margin-top:12px">
          <thead>
            <tr style="background:#1e1e3a">
              <th style="padding:10px 14px;text-align:left;color:#a0a0d0;font-size:13px;border-bottom:1px solid #2e2e4e">직원</th>
              <th style="padding:10px 14px;text-align:left;color:#a0a0d0;font-size:13px;border-bottom:1px solid #2e2e4e">역량</th>
        """
        for day in DAYS:
            table_html += f'<th style="padding:10px 14px;text-align:center;color:#a0a0d0;font-size:13px;border-bottom:1px solid #2e2e4e">{day}</th>'
        table_html += "</tr></thead><tbody>"

        for i, emp in enumerate(emps):
            row_bg = "#12122a" if i % 2 == 0 else "#0e0e22"
            bg = SKILL_COLORS[emp["skill"]]
            fg = SKILL_TEXT[emp["skill"]]
            table_html += f'<tr style="background:{row_bg}">'
            table_html += f'<td style="padding:10px 14px;color:#e0e0e0;font-weight:600;font-size:13px">{emp["name"]}</td>'
            table_html += f'<td style="padding:10px 14px"><span class="badge" style="background:{bg};color:{fg}">{emp["skill"]}</span></td>'
            for di in range(5):
                shift = assign_shift(emp, di)
                table_html += f'<td style="padding:10px 14px;text-align:center">{shift_badge(shift)}</td>'
            table_html += "</tr>"

        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

        # 주간 근무시간 차트
        st.markdown("#### 직원별 예상 주간 근무 시간")
        SHIFT_HRS = {"오전":4,"오후":4,"풀타임":8,"휴무":0}
        total_hrs = [sum(SHIFT_HRS[assign_shift(e, i)] for i in range(5)) for e in emps]
        bar_c = [SKILL_COLORS[e["skill"]] for e in emps]
        names = [e["name"] for e in emps]

        fig2, ax2 = plt.subplots(figsize=(8, 2.8))
        bars2 = ax2.bar(names, total_hrs, color=bar_c, width=0.5, edgecolor="#1a1a2e", linewidth=1.5)
        ax2.axhline(y=40, color="#d94040", linestyle="--", linewidth=1.2, label="법정 근무 한도 (40h)", alpha=0.9)
        ax2.set_ylabel("시간(h)", fontsize=10, color=CHART_FG)
        ax2.set_ylim(0, 50)
        for bar, h in zip(bars2, total_hrs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{h}h", ha="center", fontsize=10, color=CHART_FG)
        ax2.set_xticklabels(names, color=CHART_FG)
        legend2 = ax2.legend(fontsize=9, facecolor=CHART_BG, edgecolor=CHART_GRID, labelcolor=CHART_FG)
        apply_dark_style(fig2, ax2)
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("**근무 종류 설명**")
        lc = st.columns(4)
        items = [("🔵 오전","오전 4시간"),("🟣 오후","오후 4시간"),("🟢 풀타임","전일 8시간"),("⚫ 휴무","미배치")]
        for col, (label, desc) in zip(lc, items):
            col.caption(f"{label}: {desc}")

# ═══════════════════════════════════════════════
# TAB 3 — AI 분석 리포트
# ═══════════════════════════════════════════════
with tab3:
    st.markdown("#### AI 인사 관리 리포트")
    if not emps:
        st.info("직원을 먼저 등록해주세요.")
    else:
        score  = calc_score(emps)
        high_f = [e for e in emps if e["fatigue"] >= 4]
        score_color = "#5db832" if score >= 85 else ("#e8a020" if score >= 70 else "#d94040")

        col_score, col_info = st.columns([1, 3])
        with col_score:
            st.markdown(
                f'<div style="border:3px solid {score_color};border-radius:50%;'
                f'width:96px;height:96px;display:flex;flex-direction:column;'
                f'align-items:center;justify-content:center;margin:auto">'
                f'<span style="font-size:26px;font-weight:600;color:{score_color}">{score}</span>'
                f'<span style="font-size:11px;color:#aaa">최적화 점수</span></div>',
                unsafe_allow_html=True
            )
        with col_info:
            st.markdown(f"**팀 최적화 점수: {score}점**")
            if score >= 85:
                st.markdown('<div class="ok-box">✅ 팀 상태가 매우 양호합니다.</div>', unsafe_allow_html=True)
            elif score >= 70:
                st.markdown('<div class="warn-box">⚠️ 일부 직원 피로도 관리가 필요합니다.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="err-box">🚨 팀 피로도가 위험 수준입니다. 즉각적인 조치가 필요합니다.</div>', unsafe_allow_html=True)

        st.divider()

        # 파이 차트
        st.markdown("#### 역량별 인력 분포")
        skill_cnt = Counter(e["skill"] for e in emps)
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        labels = list(skill_cnt.keys())
        sizes  = list(skill_cnt.values())
        colors = [SKILL_COLORS[s] for s in labels]
        wedges, texts, autotexts = ax3.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.0f%%", startangle=90,
            wedgeprops={"edgecolor":"#1a1a2e","linewidth":2}
        )
        for t in texts:
            t.set_fontsize(10)
            t.set_color(CHART_FG)
        for at in autotexts:
            at.set_fontsize(9)
            at.set_color("#1a1a2e")
        fig3.patch.set_facecolor(CHART_BG)
        st.pyplot(fig3)
        plt.close(fig3)

        # 직원별 상세 현황
        st.markdown("#### 직원별 상태 현황")
        detail_data = []
        for emp in emps:
            hrs = sum({"오전":4,"오후":4,"풀타임":8,"휴무":0}[assign_shift(emp, i)] for i in range(5))
            detail_data.append({
                "이름":      emp["name"],
                "역량":      emp["skill"],
                "피로도":    f"{emp['fatigue']}점",
                "상태":      get_status(emp["fatigue"]),
                "주간 근무": f"{hrs}h",
                "선호 근무": emp["pref"],
            })
        st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### 🤖 경영 넛지 조언")
        st.markdown(f'<div class="nudge-box">💡 {get_nudge(emps)}</div>', unsafe_allow_html=True)

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
            '<p style="font-size:12px;color:#666;text-align:center">'
            '⚠️ 본 시스템은 알고리즘 경영 학습용입니다. 실제 인사 결정에는 관리자의 최종 판단이 필요합니다.</p>',
            unsafe_allow_html=True
        )
