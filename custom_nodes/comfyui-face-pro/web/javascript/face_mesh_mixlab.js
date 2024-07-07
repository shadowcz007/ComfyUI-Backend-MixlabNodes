import { app } from '../../../scripts/app.js'


const opts = {
  left_eyebrow: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
  right_eyebrow: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
  left_eye: [
    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7
  ],
  right_eye: [
    263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390,
    249
  ],
  inner_lip: [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14,
    87, 178, 88, 95
  ],
  outer_lip: [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84,
    181, 91, 146
  ],
  face_boundary: [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
  ],
  left_iris: [468, 469, 470, 471, 472],
  right_iris: [473, 474, 475, 476, 477],
  nose: [64, 4, 294]
}

app.registerExtension({
  name: 'Mixlab.facepro.FaceMeshMask_',
  async beforeRegisterNodeDef (nodeType, nodeData, app) {
    if (nodeData.name === 'FaceMeshMask_') {
      const orig_nodeCreated = nodeType.prototype.onNodeCreated
      nodeType.prototype.onNodeCreated = function () {
        orig_nodeCreated?.apply(this, arguments)
        // console.log(this.widgets)
        let option = this.widgets.filter(w => w.name == 'option')[0]
        // console.log(option)
        let landmarks = this.widgets.filter(w => w.name == 'landmarks')[0]
        option.callback = () => {
          landmarks.value = opts[option.value].join(' ')
        }

        this.serialize_widgets = true //需要保存参数
      }
    }
  }
})
