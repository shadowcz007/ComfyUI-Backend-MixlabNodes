const smart_connect_config_input = [
  {
    node_type: 'CLIPTextEncode',
    node_widget_name: 'text',
    inputNodeName: 'RandomPrompt',
    inputNode_output_name: 'STRING'
  },
  {
    node_type: 'CLIPTextEncode',
    node_widget_name: 'text',
    inputNodeName: 'EmbeddingPrompt',
    inputNode_output_name: 'STRING'
  },
  {
    node_type: 'CLIPTextEncode',
    node_widget_name: 'text',
    inputNodeName: 'ChinesePrompt_Mix',
    inputNode_output_name: 'prompt'
  },
  {
    node_type: 'CheckpointLoaderSimple',
    node_widget_name: 'ckpt_name',
    inputNodeName: 'CkptNames_',
    inputNode_output_name: 'ckpt_names'
  },
  {
    node_type: 'KSampler',
    node_widget_name: 'sampler_name',
    inputNodeName: 'SamplerNames_',
    inputNode_output_name: 'sampler_names'
  },
  {
    node_type: 'LoraLoaderModelOnly',
    node_widget_name: 'lora_name',
    inputNodeName: 'LoraNames_',
    inputNode_output_name: 'lora_names'
  },
  {
    node_type: 'LoadLoRA',
    node_widget_name: 'lora_name',
    inputNodeName: 'LoraNames_',
    inputNode_output_name: 'lora_names'
  },
  {
    node_type: 'Moondream',
    node_widget_name: 'image',
    inputNodeName: 'LoadImage',
    inputNode_output_name: 'IMAGE'
  },
  {
    node_type: 'TripoSRSampler_',
    node_widget_name: 'image',
    inputNodeName: 'LoadImagesToBatch',
    inputNode_output_name: 'IMAGE'
  },
  {
    node_type: 'TripoSRSampler_',
    node_widget_name: 'mask',
    inputNodeName: 'RembgNode_Mix',
    inputNode_output_name: 'masks'
  }
]

const smart_connect_config_output = [
  {
    node_type: 'LoadImage',
    node_output_name: 'IMAGE',
    outputNodeName: 'ClipInterrogator',
    outputNode_input_name: 'image'
  },
  {
    node_type: 'VAEDecode',
    node_output_name: 'IMAGE',
    outputNodeName: 'PromptImage',
    outputNode_input_name: 'images'
  },
  {
    node_type: 'VAEDecode',
    node_output_name: 'IMAGE',
    outputNodeName: 'PreviewImage',
    outputNode_input_name: 'images'
  },
  {
    node_type: 'VAEDecode',
    node_output_name: 'IMAGE',
    outputNodeName: 'SaveImage',
    outputNode_input_name: 'images'
  },
  {
    node_type: 'VAEDecode',
    node_output_name: 'IMAGE',
    outputNodeName: 'AppInfo',
    outputNode_input_name: 'IMAGE'
  },
  {
    node_type: 'VAEDecode',
    node_output_name: 'IMAGE',
    outputNodeName: 'SaveImageAndMetadata_',
    outputNode_input_name: 'images'
  },
  {
    node_type: 'Moondream',
    node_output_name: 'STRING',
    outputNodeName: 'ShowTextForGPT',
    outputNode_input_name: 'text'
  }
]

// import {
//   convertToInput,
//   getConfig,
//   isConvertableWidget
// } from '../../../extensions/core/widgetInputs.js'

const CONVERTED_TYPE = 'converted-widget'
const GET_CONFIG = Symbol()

function getConfig (widgetName) {
  const { nodeData } = this.constructor
  return (
    nodeData?.input?.required[widgetName] ??
    nodeData?.input?.optional?.[widgetName]
  )
}

function hideWidget (node, widget, suffix = '') {
  widget.origType = widget.type
  widget.origComputeSize = widget.computeSize
  widget.origSerializeValue = widget.serializeValue
  widget.computeSize = () => [0, -4] // -4 is due to the gap litegraph adds between widgets automatically
  widget.type = CONVERTED_TYPE + suffix
  widget.serializeValue = () => {
    // Prevent serializing the widget if we have no input linked
    if (!node.inputs) {
      return undefined
    }
    let node_input = node.inputs.find(i => i.widget?.name === widget.name)

    if (!node_input || !node_input.link) {
      return undefined
    }
    return widget.origSerializeValue
      ? widget.origSerializeValue()
      : widget.value
  }

  // Hide any linked widgets, e.g. seed+seedControl
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      hideWidget(node, w, ':' + widget.name)
    }
  }
}

function convertToInput (node, widget, config) {
  hideWidget(node, widget)

  const type = config[0]

  // Add input and store widget config for creating on primitive node
  const sz = node.size
  node.addInput(widget.name, type, {
    widget: { name: widget.name, [GET_CONFIG]: () => config }
  })

  for (const widget of node.widgets) {
    widget.last_y += LiteGraph.NODE_SLOT_HEIGHT
  }

  // Restore original size but grow if needed
  node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])])
}

export function smart_init () {
  LGraphCanvas.prototype._createNodeForInput = function (
    node,
    widget,
    inputNodeName,
    inputNode_slot
  ) {
    // console.log(node.pos)

    // var widget = node.widgets.filter(w => w.name === node_widget_name)[0]
    if (widget) {
      // 如果有存在的，没有连线输出的，自动连，不新建
      let input_node = null

      Array.from(app.graph.findNodesByType(inputNodeName), n => {
        var links = n.outputs.filter(o => o.name === inputNode_slot)[0].links
        //  console.log(links)
        if (!links || links?.length === 0) input_node = n
      })
      // 新建
      if (!input_node) {
        input_node = LiteGraph.createNode(inputNodeName)
        input_node.pos = [node.pos[0] - node.size[0] - 24, node.pos[1] - 48]
        app.canvas.graph.add(input_node, false)
      } else {
        input_node.pos = [node.pos[0] - node.size[0] - 24, node.pos[1] - 48]
      }

      const config = getConfig.call(node, widget.name) ?? [
        widget.type,
        widget.options || {}
      ]
      let node_slotType = config[0]
      // 如果input没有，则创建
      if (
        !node.inputs?.filter(inp => inp.name === widget.name)[0] ||
        !node.inputs
      )
        convertToInput(node, widget, config)
      input_node.connectByType(inputNode_slot, node, node_slotType)
    }
  }

  LGraphCanvas.prototype._createNodeForOutput = function (
    node,
    widget,
    outputNodeName,
    outputNode_slot
  ) {
    if (widget) {
      let output_node
      Array.from(app.graph.findNodesByType(outputNodeName), n => {
        var links = n.inputs.filter(o => o.name === outputNode_slot)[0].links
        //  console.log(links)
        if (!links || links?.length === 0) output_node = n
      })
      console.log('output_node', output_node, widget.name)

      if (!output_node) {
        // 新建
        output_node = LiteGraph.createNode(outputNodeName)
        output_node.pos = [node.pos[0] + node.size[0] + 24, node.pos[1] - 48]
        app.canvas.graph.add(output_node, false)
      } else {
        output_node.pos = [node.pos[0] + node.size[0] + 24, node.pos[1] - 48]
      }

      const config = getConfig.call(node, widget.name) ?? [
        widget.type,
        widget.options || {}
      ]
      let node_slotType = config[0]
      console.log(node_slotType, output_node, outputNode_slot)
      let type = output_node.inputs.filter(
        inp => inp.name == outputNode_slot
      )[0].type
      node.connectByType(node_slotType, output_node, type)
    }
  }
}

export function addSmartMenu (options, node) {
  let sopts = []

  for (const sc of smart_connect_config_input) {
    // 有智能推荐，则出现
    if (node.type === sc.node_type) {
      // console.log('smart',node)
      // 则出现 randomPrompt
      // CLIPTextEncode 的widget ，name== 'text'
      let node_widget_name = sc.node_widget_name
      let widget = node.widgets.filter(w => w.name === node_widget_name)[0]
      if (!widget) {
        // 控件没有，则查找inputs
        widget = node.inputs.filter(w => w.name === node_widget_name)[0]
      }

      let isLinkNull = true
      // 如果input里已经有，但是link为空
      if (node.inputs?.filter(inp => inp.name === node_widget_name)[0]) {
        isLinkNull =
          node.inputs.filter(inp => inp.name === node_widget_name)[0].link ===
          null
      }

      if (widget && isLinkNull) {
        sopts.push({
          content: sc.inputNodeName.split('_')[0] + '➡️',
          callback: () => {
            LGraphCanvas.prototype._createNodeForInput(
              node, //当前node
              widget, //当前node里需要自动连线的widget
              sc.inputNodeName, //作为input的node type
              sc.inputNode_output_name // 作为input的node的outputs的name. the input slot type of the target node
            )
          }
        })
      }
    }
  }

  for (const sc of smart_connect_config_output) {
    if (node.type === sc.node_type) {
      let node_output_name = sc.node_output_name
      const widget = node.outputs.filter(w => w.name === node_output_name)[0]

      let isLinkNull = true
      // 如果output里 link为空
      if (node.outputs?.filter(inp => inp.name === node_output_name)[0]) {
        isLinkNull =
          node.outputs.filter(inp => inp.name === node_output_name)[0].links
            ?.length === 0
        if (!node.outputs.filter(inp => inp.name === node_output_name)[0].links)
          isLinkNull = true
      }

      if (widget && isLinkNull) {
        sopts.push({
          content: '➡️' + sc.outputNodeName.split('_')[0],
          callback: () => {
            LGraphCanvas.prototype._createNodeForOutput(
              node, //当前node
              widget, //当前node里需要自动连线的widget
              sc.outputNodeName, //作为input的node type
              sc.outputNode_input_name // 作为input的node的outputs的name. the input slot type of the target node
            )
          }
        })
      }
    }
  }

  if (sopts.length > 0) options = [...sopts, null, ...options]

  return options
}
