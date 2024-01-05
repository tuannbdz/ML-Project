const goToBottomBtn = document.getElementById("goToBottomBtn")
const boxchat = document.getElementById("boxchat")

goToBottomBtn.addEventListener('click', () => {
    boxchat.scrollTop = boxchat.scrollHeight
})

boxchat.addEventListener('scroll', () => {
    if(boxchat.scrollHeight - boxchat.scrollTop < boxchat.clientHeight + 200){
        goToBottomBtn.style.display = "none"
    }
    else{
        goToBottomBtn.style.display = "flex"
    }
})

$("#requestForm").submit(function (event) {
    event.preventDefault();
    callAPI();
  });
  function callAPI() {
    const prompt = document.getElementById("prompt").value;
    // console.log(prompt);
    if (!prompt) return;
    document.getElementById("prompt").value = "";

    //USER MESSAGE
    const user_output = `
        <div class="chat">
            <div class="chat-icon">
                <i class="fa-solid fa-user fa-2xl" style="color: #1738be;"></i>
            </div>
            <div class="chat-content">
                <div class="chat-content-title">User</div>
                <div class="chat-content-text">${prompt}</div>
            </div>
        </div>
    `
    $("#boxchat").append(user_output);

    //BOT MESSAGE
    $.post("/api/model", { prompt: prompt }, (data, response) => {
      console.log('123123', prompt);
      const ans = data.ans;
      const bot_output = `
        <div class="chat">
            <div class="chat-icon">
                <i class="fa-solid fa-robot fa-2xl" style="color: #1738be;"></i>
            </div>
            <div class="chat-content">
                <div class="chat-content-title">BotChat</div>
                <div class="chat-content-text">${ans}</div>
            </div>
        </div>
    `
      $("#boxchat").append(bot_output);
      var container = $("#boxchat");
      container.scrollTop(container.prop("scrollHeight"));
    });
  }