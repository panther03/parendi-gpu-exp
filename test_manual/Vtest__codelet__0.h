#ifndef VTEST__CODELET__0_H_
#define VTEST__CODELET__0_H_

#include <vlcuda/verilated.h>

class Vtest___TOP;

//
// | LOAD (GLOBAL -> SHARED)  | COMPUTE, STORE | ....

// at TILE = 0   WORKER = 0
class Vtest___VBspCls_vtxCls__0 : public VBspCudaTileCls<Vtest___VBspCls_vtxCls__0, Vtest___TOP> {
  public:
    /* [L] */
    IData/*0:0*/ __VBspMember_clk__0; /* __VBspMember_clk__0 : test.v:1:24 */
    /* [L] */
    IData/*0:0*/ __VBspMember_ha97880fa__0; /* __VBspMember_ha97880fa__0 : test.v:5:17 */
    /* [O] */
    IData/*31:0*/ __VBspMember_TOP__DOT__test__DOT__ra__0; /* TOP->test.ra : test.v:2:18 */
    /* [I] */
    IData/*31:0*/ __VBspMember_TOP__DOT__test__DOT__rc__0; /* TOP->test.rc : test.v:4:18 */
    VL_INLINE_OPT void nbaTop(const VlTriggerVec<1> &__VBspMember_trigArg__0);
    VL_INLINE_OPT void storeGlobal(Vtest___TOP *topState);
    VL_INLINE_OPT VlTriggerVec<1> triggerEval();
    VL_INLINE_OPT void compute(Vtest___TOP *topState);
    VL_INLINE_OPT void exchangeLoad(Vtest___TOP *topState);
};

// at TILE = 1   WORKER = 0
class Vtest___VBspCls_vtxCls__1 : public VBspCudaTileCls<Vtest___VBspCls_vtxCls__1, Vtest___TOP> {
  public:
    /* [L] */
    IData/*0:0*/ __VBspMember_clk__0; /* __VBspMember_clk__0 : test.v:1:24 */
    /* [L] */
    IData/*0:0*/ __VBspMember_ha97880fa__0; /* __VBspMember_ha97880fa__0 : test.v:5:17 */
    /* [O] */
    IData/*31:0*/ __VBspMember_TOP__DOT__test__DOT__rc__0; /* TOP->test.rc : test.v:4:18 */
    /* [I] */
    IData/*31:0*/ __VBspMember_TOP__DOT__test__DOT__rb__0; /* TOP->test.rb : test.v:3:18 */
    VL_INLINE_OPT void nbaTop(const VlTriggerVec<1> &__VBspMember_trigArg__0);
    VL_INLINE_OPT void storeG
    
    jasorVec<1> triggerEval();
    VL_INLINE_OPT void compute(Vtest___TOP *topState);
    VL_INLINE_OPT void exchangeLoad(Vtest___TOP *topState);
};

// at TILE = 2   WORKER = 0
class Vtest___VBspCls_vtxCls__2 : public VBspCudaTileCls<Vtest___VBspCls_vtxCls__2, Vtest___TOP> {
  public:
    /* [L] */
    IData/*0:0*/ __VBspMember_clk__0; /* __VBspMember_clk__0 : test.v:1:24 */
    /* [L] */
    IData/*0:0*/ __VBspMember_ha97880fa__0; /* __VBspMember_ha97880fa__0 : test.v:5:17 */
    /* [O] */
    IData/*31:0*/ __VBspMember_TOP__DOT__test__DOT__rb__0; /* TOP->test.rb : test.v:3:18 */
    /* [I] */
    IData/*31:0*/ __VBspMember_TOP__DOT__test__DOT__ra__0; /* TOP->test.ra : test.v:2:18 */
    VL_INLINE_OPT void nbaTop(const VlTriggerVec<1> &__VBspMember_trigArg__0);
    VL_INLINE_OPT void storeGlobal(Vtest___TOP *topState);
    VL_INLINE_OPT VlTriggerVec<1> triggerEval();
    VL_INLINE_OPT void compute(Vtest___TOP *topState);
    VL_INLINE_OPT void exchangeLoad(Vtest___TOP *topState);
};

// at TILE = 3   WORKER = 0
class Vtest___VBspCls_vtxCls__3 : public VBspCudaTileCls<Vtest___VBspCls_vtxCls__3, Vtest___TOP> {
  public:
    /* [L] */
    IData/*0:0*/ __VBspMember_clk__0; /* __VBspMember_clk__0 : test.v:1:24 */
    /* [L] */
    IData/*0:0*/ __VBspMember_ha97880fa__0; /* __VBspMember_ha97880fa__0 : test.v:5:17 */
    /* [I] */
    IData/*31:0*/ __VBspMember_TOP__DOT__test__DOT__rc__0; /* TOP->test.rc : test.v:4:18 */
    VL_INLINE_OPT void nbaTop(const VlTriggerVec<1> &__VBspMember_trigArg__0);
    VL_INLINE_OPT void storeGlobal(Vtest___TOP *topState);
    VL_INLINE_OPT VlTriggerVec<1> triggerEval();
    VL_INLINE_OPT void compute(Vtest___TOP *topState);
    VL_INLINE_OPT void exchangeLoad(Vtest___TOP *topState);
};

class Vtest___TOP {
  public:
    Vtest___VBspCls_vtxCls__0 __VBspCls_vtxInst__0;
    Vtest___VBspCls_vtxCls__1 __VBspCls_vtxInst__1;
    Vtest___VBspCls_vtxCls__2 __VBspCls_vtxInst__2;
    Vtest___VBspCls_vtxCls__3 __VBspCls_vtxInst__3;

    void initCompute();
};

#endif // VTEST__CODELET__0_H_