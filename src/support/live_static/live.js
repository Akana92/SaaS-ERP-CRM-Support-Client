'use strict';
const $ = id => document.getElementById(id);
const page = document.body.dataset.page;
const prefix = `support-live-${page}`;
const saved = key => { try { return JSON.parse(sessionStorage.getItem(`${prefix}-${key}`)); } catch { return null; } };
const save = (key, value) => sessionStorage.setItem(`${prefix}-${key}`, JSON.stringify(value));
const el = (tag, text, className) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = String(text); if (className) node.className = className; return node; };
const empty = text => el('p', text, 'empty');
const fields = {category:'Категория',priority:'Приоритет',sentiment:'Настроение',recommended_action:'Действие'};
const names = {repeated_previous_response:'Модель повторила предыдущий ответ',context_budget_exceeded:'Превышен допустимый размер контекста — нужен оператор',Bug:'Ошибка приложения',Plans:'Тарифы',Settings:'Настройки',AccountAccess:'Доступ к аккаунту',Integration:'Интеграции',Payment:'Оплата',ServiceIncident:'Сбой сервиса',Other:'Другое',Low:'Низкий',Medium:'Средний',High:'Высокий',Critical:'Критический',Positive:'Позитивное',Neutral:'Нейтральное',Negative:'Негативное',provide_instructions:'Предоставить инструкцию',explain_plan:'Объяснить условия тарифа',check_payment:'Проверить оплату',review_access:'Проверить доступ',troubleshoot_integration:'Проверить интеграцию',request_information:'Запросить уточнение',escalate_human:'Подключить оператора',category:'Категория',priority:'Приоритет',sentiment:'Настроение',recommended_action:'Рекомендуемое действие',human_escalation:'Передача оператору',all_labels:'Все пять меток',critical_recall:'Обнаружение критических обращений',payment_confirmed_but_access_not_active:'Оплата подтверждена, доступ не активирован',payment_dispute:'Спор по оплате',access_change_requires_operator:'Изменение доступа требует оператора',security_risk:'Риск безопасности',service_outage:'Сбой сервиса',integration_requires_operator:'Интеграция требует проверки оператором',bug_requires_operator:'Ошибка требует проверки оператором',missing_or_conflicting_facts:'Недостаточные или противоречивые данные',untrusted_instruction:'Обнаружена недоверенная инструкция',model_output_invalid:'Ответ модели не прошёл проверку',model_runtime_error:'Ошибка выполнения модели',erp_unavailable:'ERP недоступна',erp_stale:'Данные ERP устарели',erp_forbidden:'Доступ к данным ERP запрещён',draft:'Черновик',approved:'Одобрено локально'};
const readable = value => value == null ? 'не определён' : (names[value] || value);
const modelName = mode => ({base:'Base',fine_tuned:'Fine-tuned'}[mode]||mode||'не указан');
const turnCount = n => `${n} ${n%100>=11&&n%100<=14?'ходов':n%10===1?'ход':n%10>=2&&n%10<=4?'хода':'ходов'}`;
const routes = {server_clarification:'Нужно уточнение',auto_answer:'Ответ клиенту',request_information:'Нужно уточнение',human_escalation:'Передано оператору',answered:'Ответ получен',needs_information:'Нужно уточнение',fallback:'Нужна помощь специалиста'};
let boot, conversation = null, state = null, busy = false;
let requestPage=1, comparisonPage=1, requestSequence=0, detailSequence=0, comparisonSequence=0, jobSequence=0;
let selectedRequest=null, selectedJob=null, pollTimer=null, requestPageIds=new Set();
let comparisonLoadFailed=false;
let queuePage=1, queueSequence=0, queueDetailSequence=0, queueSelected=null, queueDirty=false, queueSaving=false;
let queueFilters={q:'',status:'draft',priority:''}, switchAdminTab=()=>{};
let clientPending=null, clientFailed=saved('failed'), inputGuard=null, clientTimer=null, clientRequestSequence=0;
let clientFollowLatest=false, inputRejection=saved('inputRejection')||'';
const audienceValue=()=>document.querySelector('input[name=audience]:checked')?.value||'customer';
function chooseAudience(value){document.querySelectorAll('input[name=audience]').forEach(n=>n.checked=n.value===value);renderDemoExamples();}
function audienceDisabled(value){document.querySelectorAll('input[name=audience]').forEach(n=>n.disabled=value);}
const audienceName=value=>value==='employee'?'Сотрудник':value==='customer'?'Клиент':'Архив демо';
function dateText(value){const d=new Date(typeof value==='number'?value*1000:value);return Number.isNaN(d.getTime())?'Дата неизвестна':d.toLocaleString('ru-RU',{day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'});}
async function api(url, body) {
  const response = await fetch(url, {credentials:'same-origin', ...(body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})});
  const data = await response.json();
  if (!response.ok) { const detail=data.detail;const failure=new Error(typeof detail==='string'?detail:detail?.message||'Не удалось выполнить запрос. Попробуйте ещё раз.');failure.status=response.status;failure.detail=typeof detail==='object'&&detail?detail:{};throw failure; }
  return data;
}
function error(message='',source='general') { $('error').textContent=message; $('error').hidden=!message; $('error').dataset.source=source; }
function labels(value) { const box=el('div',undefined,'labels'); if(!value) return box; for(const [key,title] of Object.entries(fields)){const item=el('div',undefined,'label');item.append(el('small',title),el('strong',readable(value[key])));box.append(item);}return box; }
function technical(data) { const details=el('details',undefined,'detail');details.append(el('summary','Технические данные'),el('pre',JSON.stringify(data,null,2)));return details; }
function fillSelect(node, rows, idKey, title, selected) { node.replaceChildren(); if(!rows.length){const option=el('option','Пока нет записей');option.value='';node.append(option);} for(const row of rows){const option=el('option',title(row));option.value=row[idKey];node.append(option);} if(rows.some(r=>r[idKey]===selected))node.value=selected; }
function download(data,name){const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function setBusy(value){busy=value;$('send').disabled=value;$('message').readOnly=page==='client'?false:value;audienceDisabled(value || (page==='client'&&!!conversation));if($('new'))$('new').disabled=value;document.querySelectorAll('#history button').forEach(b=>b.disabled=value);$('status').textContent=value?'Обрабатываем запрос. Можно подождать на этой странице; повторная отправка не нужна.':'Готово';if(page==='admin')comparisonComposer();else clientControls();}
function pendingPayload(message,context){const prior=saved('pending');if(prior && prior.message===message && prior.context===context)return prior;const pending={message,context,key:crypto.randomUUID()};save('pending',pending);return pending;}
function demoExampleControls(){
  document.querySelectorAll('#demo-example-list button').forEach(button=>button.disabled=busy||$('send').disabled);
}
function renderDemoExamples(){
  const panel=$('demo-examples'),list=$('demo-example-list');if(!panel||!boot)return;
  const audience=boot.audiences.find(row=>row.id===audienceValue());
  const examples=(audience?.examples||[]).filter(row=>typeof row.label==='string'&&typeof row.message==='string'&&row.message.trim()).slice(0,3);
  panel.hidden=!examples.length;
  const key=JSON.stringify(examples);if(list.dataset.examples!==key){
    list.dataset.examples=key;list.replaceChildren();
    for(const example of examples){
      const button=el('button',example.label,'button demo-example');button.type='button';button.title=example.message;
      button.onclick=()=>{
        if(busy||$('send').disabled)return;
        const input=$('message');const draft=input.value.trim()?`${input.value}\n${example.message}`:example.message;
        if(draft.length>input.maxLength){error('Пример не добавлен: в поле вопроса недостаточно места. Сократите черновик.');return;}
        input.value=draft;
        persistDraft();input.focus();input.setSelectionRange(input.value.length,input.value.length);
      };
      list.append(button);
    }
  }
  demoExampleControls();
}
function persistDraft(){save('draft',$('message').value);save('audience',audienceValue());}
function renderClient(){
  $('messages').replaceChildren();$('analysis').replaceChildren();
  if(!clientPending&&(!conversation || !conversation.messages.length))$('messages').append(empty('Начните с вопроса. Опишите ситуацию своими словами — определим категорию и подскажем следующий шаг.'));
  for(const row of conversation?.messages||[]){const user=el('div',undefined,'bubble user');user.append(el('small','Вы'),el('div',row.message));const answer=el('div',undefined,'bubble');answer.append(el('small','Поддержка'),el('div',row.client.response),el('small',routes[row.client.status]||row.client.status));$('messages').append(user,answer);}
  if(clientPending){const user=el('div',undefined,'bubble user pending-question');user.append(el('small',pendingUserLabel(),'pending-user-label'),el('div',clientPending.message));const answer=el('div',undefined,'bubble pending-answer');answer.setAttribute('role','status');answer.append(el('span','•••','typing-dots'),el('span',pendingStatus(),'pending-status'));$('messages').append(user,answer);}
  renderClientFailure();
  const last=conversation?.messages.at(-1)?.client;
  if(last?.analysis)$('analysis').append(labels(last.analysis));
  if(last?.escalation)$('analysis').append(el('p','Обращение передано специалисту.','badge amber'));
  if(conversation){chooseAudience(conversation.audience);audienceDisabled(true);}
  $('history').replaceChildren();for(const row of boot.conversations){const button=el('button',row.title||'Обращение','nav-item'+(row.id===conversation?.id?' active':''));button.disabled=busy;button.onclick=()=>loadConversation(row.id);$('history').append(button);}
}
async function loadConversation(id){if(busy)return;try{conversation=await api(`/api/client/conversations/${encodeURIComponent(id)}`);save('conversation',id);renderClient();error();}catch(e){error(e.message);}}
function guardSeconds(){return inputGuard?.blocked_until?Math.max(0,Math.ceil(Number(inputGuard.blocked_until)-Date.now()/1000)):0;}
function clientControls(){
  if(page!=='client')return;
  const blocked=inputGuard?.allowed===false;
  $('send').disabled=busy||!inputGuard||blocked;
  demoExampleControls();
  if($('retry-failed'))$('retry-failed').disabled=busy||!inputGuard||blocked;
  const node=$('input-guard');if(!node)return;
  node.replaceChildren();node.hidden=false;
  if(!inputGuard){node.append(el('span','Проверяем доступность отправки…'));return;}
  if(blocked){const seconds=guardSeconds();node.append(el('span',`Отправка заблокирована на 5 минут. ${seconds?`Повторная проверка через ${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')}.`:'Проверяем возможность отправки…'}`));}
  else if(inputGuard.attempts_remaining<3){node.append(el('span',inputRejection||inputGuard.message||`Опишите вопрос словами. До временной блокировки осталось попыток: ${inputGuard.attempts_remaining}.`));}
  node.hidden=inputGuard.allowed&&inputGuard.attempts_remaining>=3;
}
async function refreshInputGuard(){
  try{inputGuard=await api('/api/client/input-status');if(inputGuard.allowed&&inputGuard.attempts_remaining>=3){inputRejection='';save('inputRejection','');}save('inputGuard',inputGuard);clientControls();}
  catch{const node=$('input-guard');if(node){node.hidden=false;node.replaceChildren(el('span','Не удалось проверить доступность отправки. '));const retry=el('button','Проверить снова','button quiet');retry.type='button';retry.onclick=refreshInputGuard;node.append(retry);}}
}
function pendingUserLabel(){return ['queued','running'].includes(clientPending?.status)?'Вы · отправлено':'Вы · отправляется';}
function scrollClientComposer(){
  if(!clientFollowLatest)return;
  requestAnimationFrame(()=>{if(clientFollowLatest)$('composer').scrollIntoView({block:'end',behavior:'instant'});});
}
function pendingStatus(){
  if(!clientPending)return '';
  const elapsed=Math.max(0,Math.floor((Date.now()-clientPending.started)/1000));
  const text=clientPending.status==='running'?'Готовим ответ…':clientPending.status==='queued'?'Сообщение принято. Ожидаем обработки…':'Отправляем сообщение…';
  return `${text} · ${elapsed} с`;
}
function renderClientFailure(){
  let box=$('failed-message');if(!box){box=el('div',undefined,'failed-message');box.id='failed-message';$('composer').append(box);}
  box.replaceChildren();box.hidden=!clientFailed||!!(clientFailed.conversationId&&clientFailed.conversationId!==conversation?.id);if(box.hidden)return;
  box.append(el('p','Неотправленный вопрос сохранён','demo-note'),el('p',clientFailed.message,'answer'));
  const retry=el('button','Повторить этот вопрос','button');retry.id='retry-failed';retry.type='button';retry.disabled=busy||!inputGuard||!inputGuard.allowed;
  retry.onclick=()=>{if(!busy&&inputGuard?.allowed)sendClient(clientFailed.message,clientFailed);};box.append(retry);
}
function clientFailure(message){
  if(!clientPending)return;
  clientFailed={...clientPending};save('failed',clientFailed);
  if(!$('message').value.trim()){$('message').value=clientPending.message;persistDraft();}
  clientPending=null;save('inflight',null);setBusy(false);renderClient();error(`${message} Вопрос сохранён для повторной отправки.`);
}
async function finishClient(row,expected=clientPending){
  if(!expected)return;
  const updated=!row?await api(`/api/client/conversations/${encodeURIComponent(expected.conversationId)}`):null;
  if(clientPending!==expected)return;
  if(updated)conversation=updated;
  else if(!conversation.messages.some(r=>r.request_id===row.request_id))conversation.messages.push(row);
  conversation.title=conversation.messages[0]?.message.slice(0,80)||conversation.title;
  const entry=boot.conversations.find(r=>r.id===conversation.id);if(entry)entry.title=conversation.title;
  clientPending=null;clientFailed=null;save('inflight',null);save('failed',null);save('pending',null);setBusy(false);renderClient();error();scrollClientComposer();
}
async function clientTick(){
  clearTimeout(clientTimer);clientControls();
  if(clientPending){
    document.querySelector('.pending-status')?.replaceChildren(document.createTextNode(pendingStatus()));
    if(clientPending.conversationId){
      const pending=clientPending;
      try{
        const progress=await api(`/api/client/conversations/${encodeURIComponent(pending.conversationId)}/messages/${encodeURIComponent(pending.key)}/status`);
        if(clientPending===pending){
          if(['queued','running'].includes(progress.status)){pending.status=progress.status;if(progress.created_at){const time=new Date(typeof progress.created_at==='number'?progress.created_at*1000:progress.created_at).getTime();if(Number.isFinite(time))pending.started=time;}save('inflight',pending);document.querySelector('.pending-user-label')?.replaceChildren(document.createTextNode(pendingUserLabel()));document.querySelector('.pending-status')?.replaceChildren(document.createTextNode(pendingStatus()));}
          else if(progress.status==='completed'){await finishClient();}
          else if(progress.status==='failed'){clientFailure('Обработка завершилась с ошибкой.');}
          else if(progress.status==='not_found'&&pending.recovering){conversation=await api(`/api/client/conversations/${encodeURIComponent(pending.conversationId)}`);clientFailure('Сервер не подтвердил результат предыдущей отправки. Повтор использует прежний ключ запроса.');}
        }
      }catch{if(clientPending===pending&&pending.recovering)clientFailure('Не удалось восстановить статус отправки. Повтор использует прежний ключ запроса.');}
    }
  }
  if(inputGuard?.allowed===false&&guardSeconds()===0)await refreshInputGuard();
  if(clientPending||inputGuard?.allowed===false)clientTimer=setTimeout(clientTick,1000);
}
async function sendClient(message,retry=null){
  if(busy||!inputGuard?.allowed)return;
  if(retry?.conversationId&&retry.conversationId!==conversation?.id){error('Откройте исходную беседу для повторной отправки сохранённого вопроса.');return;}
  const sequence=++clientRequestSequence;const context=conversation?.id||`new:${audienceValue()}`;
  const pending=retry?.key?retry:pendingPayload(message,context);
  clientPending={message,key:pending.key,context,conversationId:conversation?.id||null,started:Date.now(),status:'not_found'};
  save('inflight',clientPending);clientFailed=null;save('failed',null);error();
  if($('message').value.trim()===message){$('message').value='';persistDraft();}
  clientFollowLatest=true;setBusy(true);renderClient();scrollClientComposer();clientTick();
  try{
    if(!conversation){conversation=await api('/api/client/conversations',{audience:audienceValue()});save('conversation',conversation.id);boot.conversations.unshift(conversation);clientPending.conversationId=conversation.id;clientPending.context=conversation.id;save('inflight',clientPending);}
    save('pending',{message,context:conversation.id,key:pending.key});
    const row=await api(`/api/client/conversations/${encodeURIComponent(conversation.id)}/messages`,{message,idempotency_key:pending.key});
    if(sequence!==clientRequestSequence)return;
    await finishClient(row);
  }catch(e){
    if(sequence!==clientRequestSequence)return;
    if(['invalid_message','input_blocked'].includes(e.detail?.code)){
      inputGuard={...e.detail,allowed:e.detail.code!=='input_blocked'};save('inputGuard',inputGuard);
      if(e.detail.code==='invalid_message'){inputRejection=e.message;save('inputRejection',inputRejection);}
      if(!$('message').value.trim()){$('message').value=clientPending?.message||message;persistDraft();}
      clientPending=null;clientFailed=null;save('inflight',null);save('pending',null);save('failed',null);
      setBusy(false);renderClient();error();
    }else clientFailure(e.message);
    await refreshInputGuard();clientTick();
  }
}
async function clientStart(){
  const pauseFollow=()=>{if(clientPending)clientFollowLatest=false;};
  window.addEventListener('wheel',pauseFollow,{passive:true});
  window.addEventListener('touchmove',pauseFollow,{passive:true});
  window.addEventListener('pointerdown',event=>{if(!event.target.closest?.('#composer'))pauseFollow();},{passive:true});
  window.addEventListener('keydown',event=>{if(['PageUp','PageDown','Home','End','ArrowUp','ArrowDown',' '].includes(event.key)&&!event.target.closest?.('textarea,input'))pauseFollow();});
  boot=await api('/api/client/bootstrap');chooseAudience(saved('audience')||'customer');
  const selected=saved('conversation');if(selected&&boot.conversations.some(r=>r.id===selected))conversation=await api(`/api/client/conversations/${encodeURIComponent(selected)}`);
  const guard=el('div',undefined,'input-guard');guard.id='input-guard';guard.setAttribute('role','status');$('composer').prepend(guard);
  $('message').value=saved('draft')||'';renderClient();await refreshInputGuard();
  $('new').onclick=()=>{if(busy)return;conversation=null;clientFailed=null;save('failed',null);save('conversation',null);save('pending',null);$('message').value='';persistDraft();audienceDisabled(false);renderClient();clientControls();$('message').focus();};
  $('export').onclick=()=>{if(conversation)download(conversation,'support-conversation.json');else error('Сначала создайте беседу.');};
  document.querySelectorAll('input[name=audience]').forEach(n=>n.onchange=()=>{persistDraft();renderDemoExamples();});
  $('composer').onsubmit=event=>{event.preventDefault();const message=$('message').value.trim();if(message)sendClient(message);};
  const inflight=saved('inflight');
  if(inflight&&conversation?.id===inflight.conversationId){clientPending={...inflight,recovering:true};setBusy(true);renderClient();}
  else if(inflight){clientPending=inflight;clientFailure('Отправка прервалась до подтверждения беседы.');}
  clientControls();clientTick();
}
function usageText(calls){if(!calls?.length)return 'Расход не зарегистрирован';return calls.map(u=>`Вход ${u.input_tokens??'неизвестно'} · выход ${u.output_tokens??'неизвестно'} · всего ${u.total_tokens??'неизвестно'} токенов · ${u.latency_ms??'неизвестно'} мс · API ${u.api_cost??'неизвестно'} ${u.currency||'USD'}${u.complete?'':' · неполные данные'}`).join('\n');}
function comparisonComposer(){
  const stale=selectedJob?.latest_comparison_id&&selectedJob.latest_comparison_id!==selectedJob.id;
  const blocked=selectedJob&&selectedJob.status!=='completed';
  $('send').disabled=busy||comparisonLoadFailed||!!stale||!!blocked;
  $('send').textContent=selectedJob?'Продолжить диалог →':'Начать сравнение →';
  $('new-comparison').disabled=busy;
  audienceDisabled(busy||!!selectedJob||comparisonLoadFailed);
  if(selectedJob)chooseAudience(selectedJob.audience);
  $('dialogue-state').textContent=comparisonLoadFailed?'Не удалось загрузить выбранный диалог. Обновите его или начните новый.':!selectedJob?'Новый диалог · одинаковый вопрос, отдельная история каждой модели':stale?'Открыта ранняя часть диалога. Для продолжения откройте последний ход.':blocked?'Продолжение доступно после завершения обеих моделей. При ошибке начните новый диалог.':`Следующий вопрос продолжит этот диалог · ${audienceName(selectedJob.audience)}`;
  $('latest-comparison').hidden=!stale;
  demoExampleControls();
}
function comparisonHistory(record){
  const fragment=document.createDocumentFragment();const context=record.dialogue_context;
  const count=Array.isArray(context?.included_request_ids)?context.included_request_ids.length:null;
  fragment.append(el('p',count===null?'Объём включённой истории не записан.':`В контексте: ${count} предыдущих обменов${context.history_truncated?' · ранняя история сокращена':''}.`,'demo-note'));
  if(Array.isArray(record.model_history)){
    const details=el('details',undefined,'detail model-history');details.append(el('summary','История, которую получила модель'));
    if(!record.model_history.length)details.append(el('p','Первый вопрос: предыдущих обменов нет.','demo-note'));
    for(const prior of record.model_history){
      const question=el('div',undefined,'bubble user');question.append(el('small','Пользователь'),el('div',prior.message));
      const answer=el('div',undefined,'bubble');answer.append(el('small','Предыдущий публичный ответ'),el('div',prior.client?.response||'Ответ не записан'));
      details.append(question,answer);
    }
    fragment.append(details);
  }
  return fragment;
}
function comparisonReply(record,turn,mode){
  const answer=el('div',undefined,'comparison-reply');
  if(!record){
    const active=turn.status==='running'&&turn.stage===mode;
    answer.append(el('p',turn.status==='failed'?'Ответ не получен: этот ход завершился с ошибкой.':active?'Модель отвечает…':'Ожидает ответа модели.',active?'job-active':'empty'));
    return answer;
  }
  const admin=record.admin||{};const raw=admin.raw_model_result;const client=record.client||{};
  let candidate=raw;
  if(!candidate&&admin.raw_response){try{const parsed=JSON.parse(admin.raw_response.trim().replace(/<\|im_end\|>$/,'').trim());if(parsed&&typeof parsed==='object'&&!Array.isArray(parsed))candidate=parsed;}catch{/* Broken output stays visible verbatim below. */}}
  answer.append(el('small','Публичный ответ','eyebrow'),el('p',client.response||admin.response||'Публичный ответ не записан','answer'));
  const handoff=record.simulated_handoff||client.escalation||admin.server_route==='human_escalation';
  answer.append(el('p',handoff?'В рабочем чате — передача оператору':(routes[client.status]||routes[admin.server_route]||client.status||'Статус не записан'),handoff?'badge amber':'demo-note'));
  if(handoff)answer.append(el('p','Здесь показано решение: сравнение не создаёт реального обращения в очереди.','demo-note'));
  if(!raw&&candidate)answer.append(el('p','Метки сырого ответа · результат не прошёл проверку','demo-note'));
  answer.append(labels(candidate||{}));
  const details=el('details',undefined,'detail raw-answer');details.append(el('summary',raw?'Сырой ответ модели и четыре метки':'Сырой ответ модели · результат не прошёл проверку'));
  if(candidate?.suggested_response)details.append(el('p',candidate.suggested_response,'answer'));
  details.append(el('pre',admin.raw_response||JSON.stringify(raw,null,2)||'Вывод не записан'));
  details.append(el('p',`Эскалация модели: ${raw?(raw.human_escalation?'да':'нет'):'не определена'} · решение сервера: ${handoff?'в рабочем чате нужен оператор':(routes[admin.server_route]||admin.server_route||'не записано')}`,'demo-note'));
  answer.append(details,comparisonHistory(record),el('p',usageText(admin.usage_calls),'demo-note'));
  if(record.inference_profile)answer.append(el('p',`Режим: ${record.inference_profile.toUpperCase()}`,'demo-note'));
  answer.append(el('p',mode==='base'?'Base · адаптер отключён':adapterName(record.serving_info||turn.serving_info),'demo-note'));
  return answer;
}
function renderComparison(job=selectedJob){
  const box=$('compare-results');const openDetails=new Set([...box.querySelectorAll('details[open]')].map(n=>n.dataset.detailKey));box.replaceChildren();
  const turns=job?(Array.isArray(job.turns)&&job.turns.length?job.turns:[job]):[];
  for(const[mode,title]of [['base','Base'],['fine_tuned','Fine-tuned']]){
    const panel=el('section',undefined,`card comparison-chat ${mode}`);panel.setAttribute('aria-label',`Диалог ${title}`);
    const header=el('div',undefined,'comparison-chat-header');header.append(el('h2',title),el('span',turnCount(turns.length),'badge'));panel.append(header);
    if(!turns.length)panel.append(empty('Задайте первый вопрос. Здесь будет полная история этой модели.'));
    for(const[index,turn]of turns.entries()){
      const exchange=el('article',undefined,'comparison-exchange');exchange.append(el('p',`Ход ${index+1} · ${dateText(turn.created_at)}`,'demo-note'));
      const question=el('div',undefined,'bubble user');question.append(el('small','Общий вопрос'),el('div',turn.message));
      exchange.append(question,comparisonReply(turn[mode],turn.id===job.id?{...turn,stage:job.stage}:turn,mode));
      exchange.querySelectorAll('details').forEach((node,n)=>{node.dataset.detailKey=`${turn.id}-${mode}-${n}`;node.open=openDetails.has(node.dataset.detailKey);});
      panel.append(exchange);
    }
    box.append(panel);
  }
  if(job){
    const note=el('div',undefined,'comparison-context');note.append(el('strong',`${audienceName(job.audience)} · ${turnCount(turns.length)} в открытой цепочке`),el('p',`Диалог ${job.dialogue_id||job.id} · выбранный ход ${job.id}`,'demo-note'));
    const chain=el('details',undefined,'detail');chain.append(el('summary','Показанная цепочка вопросов'));
    const list=el('ol');for(const turn of turns)list.append(el('li',`${turn.message} · ${turn.id}`));chain.append(list);note.append(chain);
    if(job.error)note.append(el('p',job.error,'error'));box.prepend(note);
  }
  comparisonComposer();
}
function jobStatus(){
  if(!selectedJob)return;
  const names={queued:'В очереди',base:'Отвечает Base',fine_tuned:'Отвечает Fine-tuned',completed:'Сравнение завершено',failed:'Сравнение завершилось с ошибкой'};
  const created=new Date(selectedJob.created_at).getTime();
  const elapsed=Number.isFinite(created)?Math.max(0,Math.floor((Date.now()-created)/1000)):null;
  $('status').textContent=(names[selectedJob.stage]||selectedJob.status)+(['completed','failed'].includes(selectedJob.status)?'':elapsed===null?'':` · прошло ${elapsed} с`);
}
async function openJob(id){
  const sequence=++jobSequence;clearTimeout(pollTimer);save('comparison',id);comparisonLoadFailed=true;comparisonComposer();
  try{
    const job=await api(`/api/admin/comparisons/${encodeURIComponent(id)}`);
    if(sequence!==jobSequence)return;
    comparisonLoadFailed=false;selectedJob=job;error();renderComparison();const active=['queued','running'].includes(job.status);setBusy(active);jobStatus();
    document.querySelectorAll('#comparison-list button').forEach(n=>n.classList.toggle('selected',n.dataset.id===id));
    if(active)pollTimer=setTimeout(()=>openJob(id),document.hidden?4000:1500);
    else {await refreshState();await loadComparisons();}
  }catch(e){if(sequence!==jobSequence)return;comparisonLoadFailed=true;setBusy(false);error(`Не удалось прочитать диалог: ${e.message} Нажмите «Обновить»; черновик сохранён.`);}
}
function paging(node,result,onChange){
  node.replaceChildren();const previous=el('button','Назад','button'),next=el('button','Далее','button');previous.type=next.type='button';previous.disabled=result.page<=1;next.disabled=result.page>=result.pages;
  previous.onclick=()=>onChange(result.page-1);next.onclick=()=>onChange(result.page+1);node.append(previous,el('span',`Страница ${result.page} из ${Math.max(1,result.pages)}`,'demo-note'),next);
}
async function loadRequests(){
  const sequence=++requestSequence;
  const query=new URLSearchParams({q:$('request-search').value,audience:$('request-audience').value,category:$('request-category').value,route:$('request-route').value,page:requestPage,page_size:10});
  for(const key of ['audience','category','route'])if(!query.get(key))query.delete(key);
  save('requestFilters',{q:$('request-search').value,audience:$('request-audience').value,category:$('request-category').value,route:$('request-route').value});
  try{const data=await api(`/api/admin/requests?${query}`);if(sequence!==requestSequence)return;
    if($('error').dataset.source==='requests')error();
    requestPageIds=new Set(data.items.map(row=>row.request_id));
    if(selectedRequest&&!requestPageIds.has(selectedRequest.request_id)){detailSequence++;selectedRequest=null;save('request',null);renderTrace();}
    $('request-count').textContent=`Найдено ${data.total} · на странице ${data.items.length}`;
    const list=$('request-list');list.replaceChildren();
    if(!data.items.length)list.append(empty('Обращений не найдено. Измените фильтры или отправьте новый вопрос.'));
    for(const row of data.items){const button=el('button',undefined,'record-row'+(row.request_id===selectedRequest?.request_id?' selected':''));button.type='button';button.dataset.id=row.request_id;button.append(el('small',`${dateText(row.created_at)} · ${audienceName(row.audience)}`),el('strong',row.message),el('span',`${readable(row.analysis?.category)} · ${routes[row.server_route]||row.server_route}`,'demo-note'));button.onclick=()=>openRequest(row.request_id,true);list.append(button);}
    paging($('request-paging'),data,p=>{requestPage=p;loadRequests();});
  }catch(e){if(sequence===requestSequence)error(e.message,'requests');}
}
async function openRequest(id,fromSelection=false){
  const sequence=++detailSequence;$('trace').replaceChildren(empty('Загружаем обращение…'));
  try{const row=await api(`/api/admin/requests/${encodeURIComponent(id)}`);if(sequence!==detailSequence)return;selectedRequest=row;save('request',id);renderTrace(row);if(fromSelection&&window.matchMedia('(max-width: 850px)').matches)$('trace').scrollIntoView({block:'start',behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});document.querySelectorAll('#request-list button').forEach(n=>n.classList.toggle('selected',n.dataset.id===id));}
  catch(e){if(sequence===detailSequence)$('trace').replaceChildren(empty(e.message));}
}
async function loadComparisons(){
  const sequence=++comparisonSequence;const query=new URLSearchParams({q:$('comparison-search').value,audience:$('comparison-audience').value,page:comparisonPage,page_size:5});
  if(!query.get('audience'))query.delete('audience');
  try{const data=await api(`/api/admin/comparisons?${query}`);if(sequence!==comparisonSequence)return;
    if($('error').dataset.source==='comparisons')error();
    $('comparison-count').textContent=`Сохранено ${data.total} · на странице ${data.items.length}`;const list=$('comparison-list');list.replaceChildren();
    if(!data.items.length)list.append(el('p','Сравнений пока нет или они не соответствуют поиску.','demo-note'));
    for(const row of data.items){const button=el('button',undefined,'record-row'+(row.id===selectedJob?.id?' selected':''));button.type='button';button.dataset.id=row.id;button.append(el('small',`${dateText(row.created_at)} · ${audienceName(row.audience)} · ${{completed:'Завершено',failed:'Ошибка',running:'Выполняется',queued:'В очереди'}[row.status]||row.status}`),el('strong',row.message));button.onclick=()=>{if(busy&&selectedJob?.id!==row.id){error('Дождитесь текущего сравнения перед переключением.');return;}openJob(row.id);};list.append(button);}
    paging($('comparison-paging'),data,p=>{comparisonPage=p;loadComparisons();});
  }catch(e){if(sequence===comparisonSequence)error(e.message,'comparisons');}
}
function renderTrace(row){const box=$('trace');box.replaceChildren();if(!row){box.append(empty('Выберите обращение слева — здесь появятся ответ и шаги обработки.'));return;}
  const card=el('div',undefined,'card trace-detail');card.append(el('p',`${dateText(row.created_at)} · ${audienceName(row.audience)}`,'demo-note'),el('h2',row.message),labels(row.admin.raw_model_result),el('p',row.admin.response,'answer'),el('p',`Решение сервера: ${routes[row.admin.server_route]||row.admin.server_route}`),el('p',usageText(row.admin.usage_calls),'demo-note'));
  card.append(el('p',adapterName(row.serving_info),'demo-note'));
  const context=row.dialogue_context;
  if(context){const included=Array.isArray(context.included_request_ids)?context.included_request_ids.length:null;card.append(el('p',included===null?'Контекст: объём предыдущей переписки не зафиксирован.':`Контекст: учтено предыдущих обменов — ${included}. Один обмен — вопрос и ответ.`,'demo-note'));if(context.history_truncated)card.append(el('p','Часть ранней переписки не включена из-за ограничения контекста.','demo-note'));}
  else card.append(el('p','Контекст переписки не зафиксирован для этого обращения.','demo-note'));
  if(Array.isArray(row.model_history)){
    const history=el('details',undefined,'detail');history.append(el('summary','Переписка, которую получила модель'));
    if(!row.model_history.length)history.append(el('p','Предыдущие обмены не включались.','demo-note'));
    for(const prior of row.model_history){
      const question=el('div',undefined,'bubble user');question.append(el('small',`Пользователь · ${dateText(prior.created_at)}`),el('div',prior.message));
      const response=el('div',undefined,'bubble');response.append(el('small','Ответ поддержки, показанный пользователю'),el('div',prior.client.response));
      history.append(question,response);
    }
    card.append(history);
  }
  const modes=[...new Set((row.admin.usage_calls||[]).map(call=>call.mode).filter(Boolean))];if(modes.length)card.append(el('p',`Ответ модели: ${modes.map(modelName).join(', ')}`,'demo-note'));
  for(const[index,event]of row.admin.trace_events.entries()){const step=el('div',undefined,'step'),content=el('div');content.append(el('strong',event.description),el('p',event.error?readable(event.error):({ok:'Завершено',failed:'Ошибка',skipped:'Пропущено'}[event.status]||event.status)));step.append(el('span',index+1,'number'),content,el('small',`${event.duration_ms} мс`));card.append(step);}card.append(technical(row));box.append(card);
}
function renderCosts(){const box=$('costs');box.replaceChildren();for(const[key,title]of [['support','Обработка обращений'],['comparison','Сравнение моделей']]){const u=state.usage[key];const card=el('div',undefined,'card');card.append(el('h2',title));if(!u){card.append(el('p','Полные данные расхода недоступны.'));}else{card.append(el('p',`${u.total_tokens??'Неизвестно'} токенов`,'metric'));for(const[label,value]of [['Входные токены',u.input_tokens],['Выходные токены',u.output_tokens],['Вызовы',u.calls],['API',u.api_cost]])card.append(el('p',`${label}: ${value??'неизвестно'}${label==='API'?' '+(u.currency||'USD'):''}`));card.append(el('p',u.complete?'Расход учтён полностью':'Есть неполные данные расхода','demo-note'));}box.append(card);}}
function queueNotice(text='',failure=false){const node=$('queue-notice');node.textContent=text;node.hidden=!text;node.classList.toggle('error',failure);}
function canLeaveQueue(){
  if(!queueDirty&&!queueSaving)return true;
  queueNotice(queueSaving?'Сохраняем редакцию. Подождите завершения.':'Сначала сохраните или отмените изменения.',true);
  return false;
}
function restoreQueueFilters(){for(const[key,id]of [['q','queue-search'],['status','queue-status'],['priority','queue-priority']])$(id).value=queueFilters[key];}
async function loadQueue(){
  const sequence=++queueSequence;const list=$('queue-list');list.setAttribute('aria-busy','true');
  $('queue-count').textContent='Загружаем очередь…';
  const query=new URLSearchParams({...queueFilters,page:queuePage,page_size:10});
  for(const key of ['status','priority'])if(!query.get(key))query.delete(key);
  try{
    const data=await api(`/api/admin/queue?${query}`);if(sequence!==queueSequence)return;
    if(data.pages&&data.page>data.pages){queuePage=data.pages;return loadQueue();}
    queuePage=data.page;$('queue-count').textContent=`Найдено ${data.total} · на странице ${data.items.length}`;list.replaceChildren();
    if(!data.items.length)list.append(empty('Нет обращений по выбранным условиям. Попробуйте изменить фильтры.'));
    for(const row of data.items){
      const button=el('button',undefined,'record-row'+(row.id===queueSelected?.id?' selected':''));button.type='button';button.dataset.id=row.id;
      button.setAttribute('aria-pressed',String(row.id===queueSelected?.id));
      button.append(el('small',`${row.created_at?dateText(row.created_at):'Дата неизвестна'} · ${audienceName(row.audience)}`),el('strong',row.message),el('span',`${readable(row.priority)} · ${readable(row.status)}`,'demo-note'),el('span',readable(row.reason),'demo-note'));
      button.onclick=()=>openQueueItem(row.id);list.append(button);
    }
    paging($('queue-paging'),data,p=>{if(!canLeaveQueue())return;queuePage=p;loadQueue();});
  }catch(e){if(sequence===queueSequence){$('queue-count').textContent='Очередь не загружена';queueNotice(e.message+' Нажмите «Обновить», чтобы повторить.',true);}}
  finally{if(sequence===queueSequence)list.setAttribute('aria-busy','false');}
}
async function openQueueItem(id){
  if(id===queueSelected?.id||!canLeaveQueue())return;
  const sequence=++queueDetailSequence;queueNotice();$('queue-detail').setAttribute('aria-busy','true');
  try{
    const row=await api(`/api/admin/queue/${encodeURIComponent(id)}`);if(sequence!==queueDetailSequence)return;
    // A user may start editing the previous ticket while this request is in flight.
    if(!canLeaveQueue())return;
    queueSelected=row;renderQueueDetail();
    document.querySelectorAll('#queue-list .record-row').forEach(node=>{const selected=node.dataset.id===id;node.classList.toggle('selected',selected);node.setAttribute('aria-pressed',String(selected));});
    if(window.matchMedia('(max-width: 850px)').matches)$('queue-detail').scrollIntoView({block:'start',behavior:'instant'});
    $('queue-detail-heading').focus({preventScroll:true});
  }catch(e){if(sequence===queueDetailSequence)queueNotice(e.message,true);}
  finally{if(sequence===queueDetailSequence)$('queue-detail').setAttribute('aria-busy','false');}
}
function renderQueueDetail(){
  const box=$('queue-detail');box.replaceChildren();const row=queueSelected;
  if(!row){box.append(empty('Выберите обращение — здесь появятся контекст и редактор ответа.'));return;}
  const form=el('form',undefined,'card queue-editor');const heading=el('h2',row.message);heading.id='queue-detail-heading';heading.tabIndex=-1;
  form.append(el('p',`${row.created_at?dateText(row.created_at):'Дата неизвестна'} · ${audienceName(row.audience)} · ${readable(row.status)}`,'demo-note'),heading,labels(row.analysis||{}),el('p',`Причина передачи: ${readable(row.reason)} · Приоритет: ${readable(row.priority)}`,'queue-state'));
  const trace=el('button','Открыть шаги обработки ↗','button quiet');trace.type='button';trace.onclick=()=>{if(!canLeaveQueue())return;switchAdminTab('trace');openRequest(row.request_id,true);};form.append(trace);
  const history=el('details',undefined,'detail queue-history');history.append(el('summary','Переписка до передачи оператору'));
  if(!row.history_available)history.append(el('p','История переписки недоступна для этого обращения. Ниже сохранён исходный черновик.','demo-note'));
  else for(const turn of row.conversation||[]){
    const question=el('div',undefined,'bubble user');question.append(el('small','Пользователь'),el('div',turn.message));
    const response=el('div',undefined,'bubble');response.append(el('small','Публичный ответ поддержки'),el('div',turn.client?.response||'Ответ не записан'));
    history.append(question,response);
  }
  form.append(history);
  const draftLabel=el('label','Черновик ответа');draftLabel.htmlFor='queue-draft';const draft=el('textarea');draft.id='queue-draft';draft.rows=6;draft.value=row.draft||'';
  const statusLabel=el('label','Статус редакции');statusLabel.htmlFor='queue-review-status';const status=el('select');status.id='queue-review-status';
  for(const[value,title]of [['draft','Черновик'],['approved','Одобрено локально']]){const option=el('option',title);option.value=value;status.append(option);}status.value=row.status;
  const actions=el('div',undefined,'actions queue-editor-actions');const submit=el('button','Сохранить локально','button primary');submit.type='submit';const revert=el('button','Отменить изменения','button');revert.type='button';revert.disabled=true;submit.disabled=true;
  const note=el('p','Редакция хранится на этом ноутбуке. Сохранение и одобрение не отправляют ответ клиенту.','demo-note');note.id='queue-save-note';note.setAttribute('role','status');draft.setAttribute('aria-describedby',note.id);
  const track=()=>{queueDirty=draft.value!==(row.draft||'')||status.value!==row.status;revert.disabled=!queueDirty;submit.disabled=!queueDirty;note.textContent=queueDirty?'Есть несохранённые изменения. Ответ клиенту не отправляется.':'Изменений нет. Ответ клиенту не отправляется.';};
  draft.oninput=track;status.onchange=track;
  revert.onclick=()=>{if(queueSaving)return;queueDirty=false;queueNotice();renderQueueDetail();};
  form.onsubmit=async event=>{
    event.preventDefault();if(queueSaving||!queueDirty)return;queueSaving=true;submit.disabled=revert.disabled=true;draft.readOnly=true;status.disabled=true;note.textContent='Сохраняем редакцию…';queueNotice();
    let savedReview=false;
    try{
      const changed=await api(`/api/admin/queue/${encodeURIComponent(row.id)}/review`,{draft:draft.value,status:status.value});
      Object.assign(row,changed);queueDirty=false;savedReview=true;
    }catch(e){note.textContent=`Не удалось сохранить: ${e.message} Черновик остаётся в редакторе.`;}
    finally{queueSaving=false;draft.readOnly=false;status.disabled=false;submit.disabled=!queueDirty;revert.disabled=!queueDirty;}
    if(savedReview){renderQueueDetail();$('queue-save-note').textContent='Сохранено локально. Клиенту ничего не отправлено.';await loadQueue();}
  };
  actions.append(submit,revert);form.append(draftLabel,draft,statusLabel,status,actions,note);box.append(form);
}
async function quality(){
  const data=await api('/api/admin/evaluation');
  const box=$('quality');box.replaceChildren();
  const intro=el('div',undefined,'card');
  intro.append(el('h2','Что показала адаптация модели'));
  if(!data.dialogue_outcomes&&data.quality_note)intro.append(el('p',data.quality_note,'answer'));
  box.append(intro);
  if(!data.available){intro.append(empty('Отчёт для активной модели недоступен.'));return;}
  const outcomes=data.dialogue_outcomes;
  if(outcomes){
    intro.append(el('p','Кандидат F выбран для локального демо','badge'),
      el('p','Проверены '+outcomes.denominator+' диалогов: '+data.turns+' ответов каждой модели. Результат — AI-оценка с оговорками, без подтверждения человеком.','answer'));
    const panels=el('div',undefined,'quality-results');
    panels.setAttribute('aria-label','Результат проверки целых диалогов');
    for(const [key,title] of [['base','Base'],['fine_tuned','Fine-tuned F']]){
      const value=outcomes[key],card=el('section',undefined,'card');
      card.append(el('h3',title),el('p',value.successful+' из '+outcomes.denominator,'metric'),
        el('p','Полностью успешные диалоги'),
        el('p','С ошибкой: '+value.failed+' · Неопределённый исход: '+value.unknown,'demo-note'));
      panels.append(card);
    }
    box.append(panels);
    const caveats=el('details',undefined,'card quality-explanation');
    caveats.append(el('summary','Как считали успех и какие есть ограничения'),
      el('p','Успешным считается весь диалог: на каждом ходе верны пять меток, ответ понятен, полезен, учитывает контекст, не повторяется без необходимости и соответствует доступным фактам и правилам.','answer'),
      el('p','49% — принятый ориентир для этой версии демо. Это описательная оценка AI-судей, без подтверждения человеком. Успех в демо не означает готовность к полностью автономной поддержке.','answer'));
    if(data.unresolved)caveats.append(el('p','Остались неопределённые критерии: '+data.unresolved.criteria+' в '+data.unresolved.rows+' ответах. Эти диалоги уже имеют ошибки, поэтому в числе успешных их нет. Формальная оценка остаётся незавершённой.','answer'));
    caveats.append(el('p','В ходе эксперимента был один операционный перезапуск без заранее заданного предела ожидания. Оговорка сохранена в отчёте. Дополнительных оценок ради повышения процента не проводили.','demo-note'));
    box.append(caveats);
  }else{
    intro.append(el('p',data.split+' · '+data.n+' обращений','badge'),
      el('p','Этот сохранённый эксперимент измеряет совпадение меток. Он не заменяет оценку смысла ответа человеком.','answer'));
  }
  const labelsCard=el('div',undefined,'card');
  labelsCard.append(el('h3','Насколько верно определены метки'),
    el('p','Каждая строка считается по отдельным ответам. Все пять верных меток ещё не означают, что весь диалог решён правильно.','demo-note'));
  const table=el('table'),thead=el('thead'),head=el('tr'),tbody=el('tbody');
  for(const text of ['Метрика','Base','Fine-tuned']){const th=el('th',text);th.scope='col';head.append(th);}
  thead.append(head);table.append(thead,tbody);
  const labelNames={all_five:'Все пять меток',contract_valid:'Корректный формат ответа'};
  const score=(value,total)=>value+'/'+total+' ('+(100*value/total).toLocaleString('ru-RU',{maximumFractionDigits:1})+'%)';
  for(const metric of data.metrics){
    const row=el('tr'),label=el('th',labelNames[metric.label]||readable(metric.label));label.scope='row';
    row.append(label,el('td',score(metric.base,metric.denominator)),el('td',score(metric.fine_tuned,metric.denominator)));tbody.append(row);
  }
  const wrap=el('div',undefined,'table-wrap');wrap.append(table);labelsCard.append(wrap);
  labelsCard.append(el('p','Замороженный эксперимент выполнен в NF4. Текущие разговоры и режим запуска не пересчитывают эти метрики.','demo-note'));
  if(data.report_url?.startsWith('/admin/')){const link=el('a','Открыть подробный отчёт →','button');link.href=data.report_url;link.target='_blank';link.rel='noopener';labelsCard.append(link);}
  box.append(labelsCard);
}
function adapterName(identity){
  if(!identity?.adapter_profile)return 'Версия адаптера не сохранена';
  return ({'quality-f':'Кандидат F','dialogue-250':'Диалоговый адаптер · 250 диалогов','full-v8':'Full-v8'}[identity.adapter_profile]||identity.adapter_profile);
}
function runtime(){
  const r=state.runtime;
  $('runtime').textContent=(r.busy?'Модель занята':r.loaded?'Модель готова':'Модель ещё не загружена')+
    ' · клиент: '+modelName(r.client_mode)+' · '+adapterName(r)+
    ' · '+(r.adapter_profile==='quality-f'?'выбран для локального демо':'оценка качества — в отдельном отчёте')+
    ' · запуск: '+({'nf4':'NF4','bf16':'BF16','fp16':'FP16'}[r.inference_profile||'nf4']||r.inference_profile);
}
async function refreshState(){state=await api('/api/admin/state?include_queue=false');runtime();renderCosts();}
async function refresh(){await refreshState();await Promise.all([loadRequests(),loadComparisons(),...(saved('tab')==='queue'?[loadQueue()]:[])]);}
async function adminStart(){
  boot=await api('/api/client/bootstrap');chooseAudience(saved('audience')||'customer');$('message').value=saved('draft')||'';
  for(const key of ['Bug','Plans','Settings','AccountAccess','Integration','Payment','ServiceIncident','Other']){const option=el('option',readable(key));option.value=key;$('request-category').append(option);}
  const filters=saved('requestFilters')||{};for(const[key,id]of [['q','request-search'],['audience','request-audience'],['category','request-category'],['route','request-route']])$(id).value=filters[key]||'';
  await refresh();await quality();$('send').disabled=false;renderTrace();renderComparison();
  const switchTab=name=>{if(saved('tab')==='queue'&&name!=='queue'&&!canLeaveQueue())return;if(!['trace','compare','costs','quality','queue'].includes(name))name='trace';document.querySelectorAll('[data-view]').forEach(n=>n.hidden=n.dataset.view!==name);document.querySelectorAll('[data-tab]').forEach(n=>{n.classList.toggle('active',n.dataset.tab===name);n.setAttribute('aria-current',n.dataset.tab===name?'page':'false');if(n.dataset.tab===name)$('title').textContent=n.textContent;});save('tab',name);if(name==='queue')loadQueue();};
  switchAdminTab=switchTab;renderQueueDetail();
  let queueDebounce;
  const filterQueue=()=>{if(!canLeaveQueue()){restoreQueueFilters();return;}queueFilters={q:$('queue-search').value,status:$('queue-status').value,priority:$('queue-priority').value};queuePage=1;loadQueue();};
  $('queue-search').oninput=()=>{clearTimeout(queueDebounce);if(!canLeaveQueue()){restoreQueueFilters();return;}queueDebounce=setTimeout(filterQueue,300);};
  for(const id of ['queue-status','queue-priority'])$(id).onchange=filterQueue;
  switchTab(saved('tab')||'trace');$('tabs').onclick=event=>{const button=event.target.closest('[data-tab]');if(button)switchTab(button.dataset.tab);};
  document.querySelectorAll('input[name=audience]').forEach(n=>n.onchange=()=>{persistDraft();renderDemoExamples();});
  $('export').onclick=async()=>{try{download(await api('/api/admin/export'),'support-admin-session.json');}catch(e){error(e.message);}};
  $('refresh').onclick=async()=>{try{await refresh();if(saved('comparison'))await openJob(saved('comparison'));}catch(e){error(e.message);}};
  let debounceRequests,debounceComparisons;
  $('request-search').oninput=()=>{clearTimeout(debounceRequests);requestSequence++;debounceRequests=setTimeout(()=>{requestPage=1;loadRequests();},300);};
  for(const id of ['request-audience','request-category','request-route'])$(id).onchange=()=>{requestPage=1;loadRequests();};
  $('comparison-search').oninput=()=>{clearTimeout(debounceComparisons);comparisonSequence++;debounceComparisons=setTimeout(()=>{comparisonPage=1;loadComparisons();},300);};
  $('comparison-audience').onchange=()=>{comparisonPage=1;loadComparisons();};
  $('new-comparison').onclick=()=>{
    if(busy)return;jobSequence++;clearTimeout(pollTimer);selectedJob=null;comparisonLoadFailed=false;
    save('comparison',null);save('pending',null);error();renderComparison();setBusy(false);persistDraft();$('message').focus();
  };
  $('latest-comparison').onclick=()=>{if(!busy&&selectedJob?.latest_comparison_id)openJob(selectedJob.latest_comparison_id);};
  $('compare-form').onsubmit=async event=>{
    event.preventDefault();if(busy||$('send').disabled)return;
    const message=$('message').value.trim();if(!message)return;
    const audience=selectedJob?.audience||audienceValue();const parent=selectedJob?.id||null;
    persistDraft();const pending=pendingPayload(message,JSON.stringify({audience,parent}));error();setBusy(true);
    try{
      const result=await api('/api/admin/comparisons',{message,audience,parent_comparison_id:parent,idempotency_key:pending.key});
      save('comparison',result.id);save('pending',null);$('message').value='';persistDraft();
      selectedJob=result;comparisonLoadFailed=false;renderComparison();jobStatus();await openJob(result.id);
    }catch(e){setBusy(false);error(`${e.message} Черновик сохранён. Повторная отправка того же вопроса не создаст дубль. Если диалог уже продолжен, обновите выбранный ход.`);}
  };
  if(saved('request')&&requestPageIds.has(saved('request')))await openRequest(saved('request'));
  if(saved('comparison'))await openJob(saved('comparison'));
  setInterval(()=>{if(!document.hidden&&busy)jobStatus();},1000);
  setInterval(async()=>{if(document.hidden)return;try{const current=await api('/api/admin/state?include_queue=false');state.runtime=current.runtime;runtime();}catch{/* A manual refresh provides recovery without repeated alerts. */}},20000);
}
$('message').addEventListener('input',persistDraft);
window.addEventListener('beforeunload',event=>{if(page==='admin'&&(queueDirty||queueSaving)){event.preventDefault();event.returnValue='';}});

if(page==='client')$('message').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing&&event.keyCode!==229){event.preventDefault();if(!busy&&!$('send').disabled)$('composer').requestSubmit();}});
(page==='client'?clientStart():adminStart()).catch(e=>{error(`Не удалось загрузить сервис: ${e.message} Обновите страницу для повторной попытки.`);$('send').disabled=true;});
