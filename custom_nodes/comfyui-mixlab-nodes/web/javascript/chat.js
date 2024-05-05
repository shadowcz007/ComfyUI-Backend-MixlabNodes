async function* completion (url, messages, controller) {
  let data = {
    model: 'gpt-3.5-turbo-16k',
    messages,
    temperature: 0.6,
    stream: true
  }
  // if (imageNode) {
  //   data = { ...data, image_data: [imageNode] }
  // }

  // let controller = new AbortController()

  let response = await fetch(url, {
    method: 'POST',
    body: JSON.stringify(data),
    headers: {
      Connection: 'keep-alive',
      'Content-Type': 'application/json',
      Accept: 'text/event-stream'
    },
    signal: controller.signal
  })

  const reader = response.body.getReader()
  const decoder = new TextDecoder()

  let content = ''
  let leftover = '' // Buffer for partially read lines

  try {
    let cont = true
    while (cont) {
      let result = await reader.read()
      if (result.done) {
        break
      }

     
      // Add any leftover data to the current chunk of data
      const text = leftover + decoder.decode(result.value)

      // Check if the last character is a line break
      const endsWithLineBreak = text.endsWith('\n')

      // Split the text into lines
      let lines = text.split('\n')

      // If the text doesn't end with a line break, then the last line is incomplete
      // Store it in leftover to be added to the next chunk of data
      if (!endsWithLineBreak) {
        leftover = lines.pop()
      } else {
        leftover = '' // Reset leftover if we have a line break at the end
      }

      // Parse all sse events and add them to result
      const regex = /^(\S+):\s(.*)$/gm
      for (const line of lines) {
        const match = regex.exec(line)
        if (match) {
          result[match[1]] = match[2]
          // since we know this is llama.cpp, let's just decode the json in data
          if (result.data) {
            result.data = JSON.parse(result.data)
            // console.log('#result.data',result.data)
            
            content += result.data.choices[0].delta?.content||''

            // yield
            yield result

            // if we got a stop token from server, we will break here
            if (result.data.choices[0].finish_reason=="stop") {
              if (result.data.generation_settings) {
                // generation_settings = result.data.generation_settings;
              }
              cont = false
              break
            }
          }
        }
      }
    }
  } catch (e) {
    console.error('llama error: ', e)
    throw e
  } finally {
    controller.abort()
  }

  return content
  // return (await response.json()).content
}

export async function completion_ (url, messages, controller, callback) {
  let request = await completion(url, messages, controller)
  for await (const chunk of request) {
    
    let content=chunk.data.choices[0].delta.content||""
    if(chunk.data.choices[0].role=="assistant"){
      //开始
      content=""
    }

    if (callback) callback(content)
  }
}
