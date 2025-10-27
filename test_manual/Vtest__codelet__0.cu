#include "Vtest__codelet__0.h"

// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// CUDA tile implementation

void Vtest___TOP::initCompute() {
    // Lifted from Vtest___VBspCls_vtxClsInit__0::compute()
    IData __VBspMember_TOP__DOT__test__DOT__ra__0 = 0U;
    IData __VBspMember_TOP__DOT__test__DOT__rc__0 = 0U;

    VL_NULL_CHECK(this->__VBspCls_vtxInst__0, "test.v", 6)->__VBspMember_TOP__DOT__test__DOT__ra__0 
    = __VBspMember_TOP__DOT__test__DOT__ra__0;
    VL_NULL_CHECK(this->__VBspCls_vtxInst__2, "test.v", 7)->__VBspMember_TOP__DOT__test__DOT__ra__0 
        = __VBspMember_TOP__DOT__test__DOT__ra__0;
    VL_NULL_CHECK(this->__VBspCls_vtxInst__0, "test.v", 6)->__VBspMember_TOP__DOT__test__DOT__rc__0 
        = __VBspMember_TOP__DOT__test__DOT__rc__0;
    VL_NULL_CHECK(this->__VBspCls_vtxInst__1, "test.v", 8)->__VBspMember_TOP__DOT__test__DOT__rc__0 
        = __VBspMember_TOP__DOT__test__DOT__rc__0;
    VL_NULL_CHECK(this->__VBspCls_vtxInst__3, "test.v", 5)->__VBspMember_TOP__DOT__test__DOT__rc__0 
        = __VBspMember_TOP__DOT__test__DOT__rc__0;
}

VL_INLINE_OPT void Vtest___VBspCls_vtxCls__0::nbaTop(const VlTriggerVec<1> &__VBspMember_trigArg__0) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__0::nbaTop\n") );
    // Body
    IData/*31:0*/ __VBspMember_TOP__DOT____Vdly__test__DOT__ra__0;
    // begin nba computation
    if (__VBspMember_trigArg__0.at(0U)) {
        __VBspMember_TOP__DOT____Vdly__test__DOT__ra__0 
            = this->__VBspMember_TOP__DOT__test__DOT__ra__0;
        __VBspMember_TOP__DOT____Vdly__test__DOT__ra__0 
            = ((IData)(1U) + (this->__VBspMember_TOP__DOT__test__DOT__ra__0 
                              + this->__VBspMember_TOP__DOT__test__DOT__rc__0));
        this->__VBspMember_TOP__DOT__test__DOT__ra__0 
            = __VBspMember_TOP__DOT____Vdly__test__DOT__ra__0;
    }
}

VL_INLINE_OPT void Vtest___VBspCls_vtxCls__0::storeGlobal(Vtest___TOP *topState) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__0::storeGlobal\n") );
    // Body
    (topState->__VBspCls_vtxInst__2)->__VBspMember_TOP__DOT__test__DOT__ra__0 
        = this->__VBspMember_TOP__DOT__test__DOT__ra__0;
}

VL_INLINE_OPT void Vtest___VBspCls_vtxCls__0::exchangeLoad(Vtest___TOP *topState) {
    this->__VBspMember_TOP__DOT__test__DOT__rc__0 = (topState->__VBspCls_vtxInst__0)->__VBspMember_TOP__DOT__test__DOT__rc__0;
}

VL_INLINE_OPT VlTriggerVec<1> Vtest___VBspCls_vtxCls__0::triggerEval() {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__0::triggerEval\n") );
    // Body
    VlTriggerVec<1> __VBspMember_actTrig__0;
    __VBspMember_actTrig__0.clear();
    while (__VBspMember_actTrig__0.empty()) {
        // active region computation
        __VBspMember_actTrig__0.set(0U, ((IData)(this->__VBspMember_clk__0) 
                                         & (~ (IData)(this->__VBspMember_ha97880fa__0))));
        this->__VBspMember_ha97880fa__0 = this->__VBspMember_clk__0;
        if (__VBspMember_actTrig__0.empty()) {
            this->__VBspMember_clk__0 = (1U & (~ (IData)(this->__VBspMember_clk__0)));
        }
    }
    return (__VBspMember_actTrig__0);
}

VL_INLINE_OPT void Vtest___VBspCls_vtxCls__0::compute(Vtest___TOP *topState) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__0::compute\n") );
    // Body
    this->storeGlobal(topState);
    this->nbaTop(this->triggerEval());
}



VL_INLINE_OPT void Vtest___VBspCls_vtxCls__1::nbaTop(const VlTriggerVec<1> &__VBspMember_trigArg__0) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__1::nbaTop\n") );
    // Body
    IData/*31:0*/ __VBspMember_TOP__DOT____Vdly__test__DOT__rc__0;
    // begin nba computation
    if (__VBspMember_trigArg__0.at(0U)) {
        __VBspMember_TOP__DOT____Vdly__test__DOT__rc__0 
            = this->__VBspMember_TOP__DOT__test__DOT__rc__0;
        __VBspMember_TOP__DOT____Vdly__test__DOT__rc__0 
            = (this->__VBspMember_TOP__DOT__test__DOT__rc__0 
               ^ (this->__VBspMember_TOP__DOT__test__DOT__rb__0 
                  >> 4U));
        this->__VBspMember_TOP__DOT__test__DOT__rc__0 
            = __VBspMember_TOP__DOT____Vdly__test__DOT__rc__0;
    }
}

VL_INLINE_OPT void Vtest___VBspCls_vtxCls__1::storeGlobal(Vtest___TOP *topState) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__1::storeGlobal\n") );
    // Body
    (topState->__VBspCls_vtxInst__0)->__VBspMember_TOP__DOT__test__DOT__rc__0 
        = this->__VBspMember_TOP__DOT__test__DOT__rc__0;
    (topState->__VBspCls_vtxInst__3)->__VBspMember_TOP__DOT__test__DOT__rc__0 
        = this->__VBspMember_TOP__DOT__test__DOT__rc__0;
}

VL_INLINE_OPT void Vtest___VBspCls_vtxCls__1::exchangeLoad(Vtest___TOP *topState) {
    // Body
    this->__VBspMember_TOP__DOT__test__DOT__rb__0 = (topState->__VBspCls_vtxInst__1)->__VBspMember_TOP__DOT__test__DOT__rb__0;
}

VL_INLINE_OPT VlTriggerVec<1> Vtest___VBspCls_vtxCls__1::triggerEval() {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__1::triggerEval\n") );
    // Body
    VlTriggerVec<1> __VBspMember_actTrig__0;
    __VBspMember_actTrig__0.clear();
    while (__VBspMember_actTrig__0.empty()) {
        // active region computation
        __VBspMember_actTrig__0.set(0U, ((IData)(this->__VBspMember_clk__0) 
                                         & (~ (IData)(this->__VBspMember_ha97880fa__0))));
        this->__VBspMember_ha97880fa__0 = this->__VBspMember_clk__0;
        if (__VBspMember_actTrig__0.empty()) {
            this->__VBspMember_clk__0 = (1U & (~ (IData)(this->__VBspMember_clk__0)));
        }
    }
    return (__VBspMember_actTrig__0);
}

VL_INLINE_OPT void Vtest___VBspCls_vtxCls__1::compute(Vtest___TOP *topState) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__1::compute\n") );
    // Body
    this->storeGlobal(topState);
    this->nbaTop(this->triggerEval());
}



VL_INLINE_OPT void Vtest___VBspCls_vtxCls__2::nbaTop(const VlTriggerVec<1> &__VBspMember_trigArg__0) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__2::nbaTop\n") );
    // Body
    IData/*31:0*/ __VBspMember_TOP__DOT____Vdly__test__DOT__rb__0;
    // begin nba computation
    if (__VBspMember_trigArg__0.at(0U)) {
        __VBspMember_TOP__DOT____Vdly__test__DOT__rb__0 
            = this->__VBspMember_TOP__DOT__test__DOT__rb__0;
        __VBspMember_TOP__DOT____Vdly__test__DOT__rb__0 
            = (this->__VBspMember_TOP__DOT__test__DOT__ra__0 
               ^ (this->__VBspMember_TOP__DOT__test__DOT__ra__0 
                  << 0xfU));
        this->__VBspMember_TOP__DOT__test__DOT__rb__0 
            = __VBspMember_TOP__DOT____Vdly__test__DOT__rb__0;
    }
}

VL_INLINE_OPT void Vtest___VBspCls_vtxCls__2::storeGlobal(Vtest___TOP *topState) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__2::storeGlobal\n") );
    // Body
    (topState->__VBspCls_vtxInst__1)->__VBspMember_TOP__DOT__test__DOT__rb__0 
        = this->__VBspMember_TOP__DOT__test__DOT__rb__0;
}

VL_INLINE_OPT void Vtest___VBspCls_vtxCls__2::exchangeLoad(Vtest___TOP *topState) {
    // Body
    this->__VBspMember_TOP__DOT__test__DOT__ra__0 = 
        (topState->__VBspCls_vtxInst__2)->__VBspMember_TOP__DOT__test__DOT__ra__0;
}

VL_INLINE_OPT VlTriggerVec<1> Vtest___VBspCls_vtxCls__2::triggerEval() {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__2::triggerEval\n") );
    // Body
    VlTriggerVec<1> __VBspMember_actTrig__0;
    __VBspMember_actTrig__0.clear();
    while (__VBspMember_actTrig__0.empty()) {
        // active region computation
        __VBspMember_actTrig__0.set(0U, ((IData)(this->__VBspMember_clk__0) 
                                         & (~ (IData)(this->__VBspMember_ha97880fa__0))));
        this->__VBspMember_ha97880fa__0 = this->__VBspMember_clk__0;
        if (__VBspMember_actTrig__0.empty()) {
            this->__VBspMember_clk__0 = (1U & (~ (IData)(this->__VBspMember_clk__0)));
        }
    }
    return (__VBspMember_actTrig__0);
}

VL_INLINE_OPT void Vtest___VBspCls_vtxCls__2::compute(Vtest___TOP *topState) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__2::compute\n") );
    // Body
    this->storeGlobal(topState);
    this->nbaTop(this->triggerEval());
}



VL_INLINE_OPT void Vtest___VBspCls_vtxCls__3::nbaTop(const VlTriggerVec<1> &__VBspMember_trigArg__0) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__3::nbaTop\n") );
    // Body
    // begin nba computation
    if (__VBspMember_trigArg__0.at(0U)) {
        if (VL_UNLIKELY((1U == this->__VBspMember_TOP__DOT__test__DOT__rc__0))) {
            VL_FINISH_MT("test.v", 10, "");
        }
    }
}

VL_INLINE_OPT void Vtest___VBspCls_vtxCls__3::storeGlobal(Vtest___TOP *topState) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__3::storeGlobal\n") );
}

VL_INLINE_OPT void Vtest___VBspCls_vtxCls__3::exchangeLoad(Vtest___TOP *topState) {
    // Body
    this->__VBspMember_TOP__DOT__test__DOT__rc__0 = (topState->__VBspCls_vtxInst__3)->__VBspMember_TOP__DOT__test__DOT__rc__0;
}

VL_INLINE_OPT VlTriggerVec<1> Vtest___VBspCls_vtxCls__3::triggerEval() {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__3::triggerEval\n") );
    // Body
    VlTriggerVec<1> __VBspMember_actTrig__0;
    __VBspMember_actTrig__0.clear();
    while (__VBspMember_actTrig__0.empty()) {
        // active region computation
        __VBspMember_actTrig__0.set(0U, ((IData)(this->__VBspMember_clk__0) 
                                         & (~ (IData)(this->__VBspMember_ha97880fa__0))));
        this->__VBspMember_ha97880fa__0 = this->__VBspMember_clk__0;
        if (__VBspMember_actTrig__0.empty()) {
            this->__VBspMember_clk__0 = (1U & (~ (IData)(this->__VBspMember_clk__0)));
        }
    }
    return (__VBspMember_actTrig__0);
}

VL_INLINE_OPT void Vtest___VBspCls_vtxCls__3::compute(Vtest___TOP *topState) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vtest___VBspCls_vtxCls__3::compute\n"); );
    // Body
    this->storeGlobal(topState);
    this->nbaTop(this->triggerEval());
}
