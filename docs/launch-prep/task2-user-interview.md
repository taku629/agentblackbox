# Task 2: ユーザーインタビュー用スクリプト

---

## 質問リスト JP

> 対象: AIエージェントを本番運用している / したいエンジニア・PM
> 想定時間: 30分 / 重要度順

### セクション A: 現状把握（5分）

**Q1.** 今どんなAIエージェントを作っていますか？どんなフレームワークやAPIを使っていますか？
> フォロー: そのエージェント、すでに本番に出ていますか？それともまだ開発・検証段階ですか？

**Q2.** エージェントを本番に出すまでの流れを教えてもらえますか？テストから本番デプロイまで、どんなステップがありますか？
> フォロー: そのプロセスで「ここが一番大変だった」というポイントはどこですか？

---

### セクション B: 「怖い」の正体（10分）

**Q3.** エージェントを本番に出すとき、一番不安に感じる瞬間はどんなときですか？具体的なエピソードがあれば教えてください。
> フォロー: その不安は「技術的な不確かさ」からですか？それとも「何をしているかわからない」という透明性の問題からですか？

**Q4.** 本番でエージェントが予期しない動作をしたとき、どうやって原因を追いましたか？そのとき何が一番困りましたか？
> フォロー: 「もしこのデータ/ログがあれば解決が早かった」というものはありましたか？

**Q5.** エージェントの動作コストや、APIの使用量を把握するために、今どんな方法を使っていますか？
> フォロー: コストが予想より高くなったことはありますか？そのとき何をしましたか？

**Q6.** エージェントが扱うデータのなかに、個人情報や機密情報が含まれることはありますか？そのリスクに対してどう対応していますか？

---

### セクション C: 既存ツールへの不満（8分）

**Q7.** エージェントの監視・ログ・デバッグに、現在どんなツールを使っていますか？（LangSmith, Langfuse, Helicone, 自前ログ など）
> フォロー: そのツールを選んだ理由は何ですか？

**Q8.** 今使っているツールで「ここが不満」「ここが足りない」と感じる点を率直に教えてください。
> フォロー: それが解決されたとしたら、どのくらい助かりますか？

**Q9.** 外部のSaaSにエージェントのログ・プロンプト・レスポンスを送ることに対して、社内や顧客から制約を受けたことはありますか？

---

### セクション D: 理想と支払い意欲（5分）

**Q10.** 理想の「AIエージェント可観測性ツール」を想像すると、どんな機能が絶対に必要ですか？何が「あれば嬉しい」ですか？

**Q11.** 今の課題が解決される可観測性ツールがあったとして、月額いくらまでなら払えますか？（個人/チーム/企業で変わるなら教えてください）
> フォロー: その金額の判断基準は何ですか？（節約できるコスト・時間・リスク回避など）

**Q12.** ローカルにデータを保存するだけで、外部に何も送らないツールがあれば、それは「魅力的」に映りますか？それとも「共有ダッシュボード」や「チームでの連携」のほうが重要ですか？

---

### セクション E: クロージング（2分）

**Q13.** エージェントの可観測性・監視まわりで、業界全体として「まだ誰も解決していない」と感じる問題はありますか？

**Q14.** 今日話してくれた内容以外で、AIエージェントを本番で運用するうえで「これが一番のボトルネック」だと思うことを1つ挙げるとしたら何ですか？

---

## 質問リスト EN

> Target: Engineers / PMs building or operating AI agents in production
> Duration: 30 minutes / Ordered by priority

### Section A: Context Setting (5 min)

**Q1.** Can you walk me through what AI agents you're currently building or running? What frameworks or APIs are you using?
> Follow-up: Are those agents already in production, or still in development/staging?

**Q2.** What does your process look like for getting an agent from development to production? What are the key steps?
> Follow-up: What's the hardest part of that process right now?

---

### Section B: The Fear Factor (10 min)

**Q3.** What's the moment you feel most anxious about shipping an AI agent to production? Can you walk me through a specific experience?
> Follow-up: Is that anxiety more about technical correctness, or about not knowing what the agent is actually doing?

**Q4.** Tell me about a time something went wrong with an agent in production. How did you debug it? What made it hard?
> Follow-up: What data or logs, if you'd had them, would have made that faster to resolve?

**Q5.** How do you currently track the cost and token usage of your agents? What's your process there?
> Follow-up: Have you ever been surprised by a cost spike? What did you do?

**Q6.** Do the agents you work with ever handle PII or sensitive data — like user emails, payment info, or proprietary content? How do you manage that risk today?

---

### Section C: Pain Points with Existing Tools (8 min)

**Q7.** What tools do you currently use for agent monitoring, logging, or debugging? (LangSmith, Langfuse, Helicone, custom logs, etc.)
> Follow-up: What made you choose those over the alternatives?

**Q8.** What do you find most frustrating or limiting about those tools?
> Follow-up: If that frustration were resolved, how much would it change your workflow?

**Q9.** Have you ever faced restrictions — from your company, clients, or compliance teams — on sending agent logs, prompts, or responses to a third-party service?
> Follow-up: How did you work around that, if at all?

---

### Section D: Ideal Solution and Willingness to Pay (5 min)

**Q10.** If you could design the perfect AI agent observability tool from scratch, what would it absolutely have to do? What would be nice-to-have?

**Q11.** If a tool existed that solved the problems you've described, what would you be willing to pay for it per month? (Personal / team / enterprise — whatever applies to you.)
> Follow-up: What would justify that price? What's the value you'd be getting?

**Q12.** A tool that stores everything locally — no cloud, no account — does that sound appealing, or is shared team access and collaboration more important to you?

---

### Section E: Closing (2 min)

**Q13.** Is there a problem in AI agent observability that you feel nobody has really solved yet?

**Q14.** Outside of what we've talked about, what's the single biggest bottleneck for you right now in running AI agents in production?

---

## 進行台本 EN

### 30-Minute Interview Script

---

#### Phase 1: Icebreak & Context (0–5 min)

**[Opening — set the tone]**

> "Hi [Name], thanks so much for making time today. I really appreciate it.
> Just to set expectations: this is a research interview, not a pitch. I'm trying to understand your experience — there are no right or wrong answers, and nothing you say will hurt my feelings.
> I'll be taking notes, and if it's okay with you, I'd like to record this so I can refer back later. Is that alright?"

**[Confirm recording consent, then:]**

> "Great. Let's start with a bit of context so I understand where you're coming from."

**Questions:**
- "What's your role, and what are you working on these days in the AI space?"
- "Are you currently running any AI agents in production, or are you more in the build/experiment phase?"

**[Transition]**

> "Got it. So you're [in production / building toward production]. That's exactly the kind of experience I want to dig into. Let me ask you about the production side of things."

---

#### Phase 2: The Core (5–20 min)

**[Open the problem space]**

> "I want to understand what it's actually like to ship and run an AI agent. Can you walk me through the last time you were close to putting an agent into production — what was going through your mind?"

**[Active listening prompts — use these to keep them talking:]**
- "That's interesting — can you say more about that?"
- "What did you do when that happened?"
- "How often does that come up?"
- "What would the ideal version of that have looked like?"

**[Core questions — pick based on flow, don't read robotically:]**

1. "What's the moment you feel most anxious about shipping an agent? Walk me through a specific time."
2. "Tell me about a debugging situation that was harder than it should have been. What made it hard?"
3. "How do you currently track costs and usage? Has that ever surprised you?"
4. "Does your agent handle any sensitive data — user info, payment data, internal docs? How do you manage that?"
5. "What tools are you using for observability or monitoring? What do you like and dislike about them?"
6. "Have you ever run into restrictions on sending agent data to a third-party service?"

**[Probe for willingness to pay]**

> "You mentioned [X pain point]. Hypothetically — if a tool solved that specific problem, would that be worth paying for? What would a fair price look like?"

**[Transition]**

> "This is really helpful. I want to shift gears slightly and get your reaction to a specific direction we're exploring."

---

#### Phase 3: Solution Reaction & Closing (20–30 min)

**[Introduce AgentBlackBox — keep it brief, it's a probe not a pitch]**

> "We're building a tool called AgentBlackBox. The idea is simple: it's a local flight recorder for AI agents. You add a one-line decorator, and every LLM call, tool invocation, cost, and error gets recorded to a local SQLite file on your machine. Nothing goes to any external server. There's a web dashboard for replaying sessions. Does anything about that resonate with what you've described?"

**[Listen carefully. Then ask:]**

- "What's your immediate reaction to that?"
- "What would make you actually try it?"
- "What would make you NOT use it — what's the thing that would be a dealbreaker?"
- "Is there something you'd want it to do that I haven't mentioned?"

**[Willingness-to-pay probe if not covered earlier]**

> "If this tool existed and worked exactly as I described — local-only, one decorator, full replay — what price point would feel right? What would change if it also had a cloud option for team sharing?"

**[Closing]**

> "This has been incredibly useful. Last question: is there anything about building or running AI agents in production that we haven't talked about, that you think is really important?"

> "I really appreciate your time. With your permission, I'd love to stay in touch — if we build something based on this research, I'd love to get your feedback early. Would that be okay?"

> "Thanks again, [Name]. I'll send you a quick follow-up note. Have a great rest of your day."

---

## サンクスメール JP/EN

### サンクスメール JP

**件名:** インタビューのお礼 — AgentBlackBox

[お名前] さん

本日はお時間をいただきありがとうございました。
AIエージェントの本番運用における課題について、とても参考になるお話を聞かせていただきました。

いただいたフィードバックは、AgentBlackBox の開発に直接反映させていただきます。
近日中にベータ版をリリース予定です。優先的にご案内できるよう、メーリングリストに登録させてください。

登録はこちら（30秒）: {BETA_SIGNUP_URL}

もし今後も意見交換させていただけるようであれば、ぜひお声がけください。
今後ともよろしくお願いいたします。

{YOUR_NAME}
AgentBlackBox

---

### サンクスメール EN

**Subject:** Thank you for the interview — AgentBlackBox

Hi [Name],

Thank you for taking the time to chat today — your perspective on running AI agents in production was genuinely eye-opening.

Your feedback will directly shape how we build AgentBlackBox. We're planning a beta release soon, and I'd love to keep you in the loop as a priority tester and advisor.

If you're open to it, you can sign up here (30 seconds): {BETA_SIGNUP_URL}

Feel free to reach out anytime — and if you have colleagues facing the same challenges, I'd love an introduction.

Thanks again,

{YOUR_NAME}
AgentBlackBox
