<h1 dir="rtl" align="right">با <bdi dir="ltr">Veridist</bdi> چه کارهایی می‌توانید انجام دهید؟</h1>

<p dir="rtl" align="right"><a href="capability-guide.md">English</a> | <a href="capability-guide.fa.md">فارسی</a> | <a href="capability-guide.de.md">Deutsch</a></p>

<p dir="rtl" align="right">این راهنما دربارهٔ نسخهٔ ۱٫۰٫۱ است. کمک می‌کند بدانید چه داده‌ای می‌توانید به برنامه بدهید، چه نتیجه‌ای بگیرید و چه کارهایی هنوز پشتیبانی نمی‌شوند.</p>

<h2 dir="rtl" align="right">می‌خواهم زمان تا وقوع یک رویداد را بررسی کنم</h2>

<p dir="rtl" align="right">تحلیل طول عمر فقط برای دستگاه نیست. همین ساختار داده می‌تواند زمان تا خرابی، عود بیماری، نکول، نخستین خسارت بیمه، ریزش مشتری، تبدیل یا هر رویداد روشن دیگری را توصیف کند. اگر رویداد تا پایان بازهٔ مشاهده رخ نداده باشد، آن مشاهده سانسورشده از راست است.</p>

<div dir="rtl" align="right">

| حوزه | نمونهٔ رخداد دقیق | نمونهٔ مشاهدهٔ سانسورشده از راست |
| --- | --- | --- |
| قابلیت اطمینان و تولید | یاتاقان پس از ۴۰۰ ساعت خراب شد | یاتاقان پس از ۴۰۰ ساعت هنوز کار می‌کرد |
| سلامت و پژوهش بقا | بیمار پس از ۳۰ روز دوباره بستری شد | تا روز سی‌ام بستری دوباره رخ نداد |
| اعتبار و بیمه | نکول یا نخستین خسارت در دورهٔ پیگیری رخ داد | تا پایان بررسی نکول یا خسارتی ثبت نشد |
| محصول دیجیتال | مشتری ریزش کرد یا تبدیل انجام شد | مشتری تا پایان بازه فعال ماند و رویداد رخ نداد |
| عملیات | تعمیر، تحویل یا خدمت کامل شد | فرایند هنگام پایان جمع‌آوری داده هنوز باز بود |

</div>

<p dir="rtl" align="right"><bdi dir="ltr">Veridist</bdi> زمانی می‌تواند یک توزیع آماری را به این زمان‌ها <strong>برازش</strong><sup id="fnref-fitting"><a href="#fn-fitting">۱</a></sup> دهد که فرض‌های مدل و قرارداد ورودی فعلی برقرار باشند. مدل‌های نمایی، وایبول کمینه و لگ‌نرمال با مکان ثابت صفر، برای رخدادهای دقیق و سانسور مستقل از راست در دسترس‌اند. مسیر فایل <bdi dir="ltr">CSV</bdi> محدودتر است: فقط مدل نماییِ نرخ‌محور را از قالب دو ستونیِ مشخص برازش می‌دهد.</p>

<p dir="rtl" align="right">در کشف تقلب و امنیت سایبری معمولاً پرسش کمی متفاوت است: آیا یک مبلغ، فاصلهٔ زمانی یا زمان پاسخ در مقایسه با توزیع مرجع غیرعادی است؟ اگر پارامترهای قابل‌دفاع از قبل موجود باشند، <bdi dir="ltr">Veridist</bdi> می‌تواند برای خانواده‌های پشتیبانی‌شده لگاریتم چگالی، احتمال دنباله و صدک را محاسبه کند. این مقادیر می‌توانند ورودی یک سامانهٔ تشخیص جداگانه و اعتبارسنجی‌شده باشند؛ خود بسته سامانهٔ کامل تشخیص تقلب را آموزش یا اجرا نمی‌کند.</p>

<div dir="rtl" align="right">

| مدل | چه رفتاری را توصیف می‌کند؟ | ورودی |
| --- | --- | --- |
| نمایی | نرخ خرابی ثابت در طول زمان | فایل CSV با قالب مشخص یا دادهٔ آماده‌شده در پایتون |
| وایبول کمینه | نرخ خرابی کاهشی، ثابت یا افزایشی، بسته به پارامتر شکل | دادهٔ آماده‌شده در پایتون |
| لگ‌نرمال | زمان‌های مثبت که لگاریتم آن‌ها از مدل نرمال پیروی می‌کند | دادهٔ آماده‌شده در پایتون |

</div>

<p dir="rtl" align="right">برای شروع با فایل CSV، فعلاً فقط مدل نمایی در دسترس است. فایل باید با کدگذاری <bdi dir="ltr">UTF-8</bdi> ذخیره شود و دقیقاً دو ستون <code dir="ltr">time,event_observed</code> به همین ترتیب داشته باشد. ستون اول مدت مشاهده است. در ستون دوم، عدد ۱ یعنی خرابی دیده شده و عدد ۰ یعنی تا پایان مشاهده خرابی دیده نشده است. برنامه قالب فایل را حدس نمی‌زند.</p>

<h2 dir="rtl" align="right">اگر بعضی دستگاه‌ها هنوز خراب نشده باشند چه؟</h2>

<p dir="rtl" align="right">اطلاعات آن‌ها هم قابل استفاده است. مثلاً اگر پمپی پس از ۱۰۰ ساعت هنوز کار می‌کند، می‌دانیم عمر آن بیشتر از ۱۰۰ ساعت بوده؛ حتی اگر زمان خرابی نهایی را ندانیم. این مشاهده را <strong>دادهٔ سانسورشده از راست</strong><sup id="fnref-censoring"><a href="#fn-censoring">۲</a></sup> می‌نامیم. هر سه مدل بالا این داده را می‌پذیرند.</p>

<p dir="rtl" align="right">روش فعلی فرض می‌کند پایان مشاهده به زمان خرابیِ هنوز مشاهده‌نشده وابسته نیست. مثلاً خارج‌کردن پمپ‌ها از مطالعه به‌دلیل نشانه‌های خرابی قریب‌الوقوع می‌تواند این فرض را نقض کند. برنامه نمی‌تواند درست‌بودن این فرض را از روی داده اثبات کند.</p>

```mermaid
timeline
    title دو مشاهده از عمر پمپ
    0 ساعت : شروع مشاهده
    100 ساعت : پمپ اول خراب شد
    100 ساعت : مشاهدهٔ پمپ دوم پایان یافت؛ خرابی آن دیده نشد
```

<p dir="rtl" align="right">در نمودار، زمان خرابی پمپ اول معلوم است. دربارهٔ پمپ دوم فقط می‌دانیم دست‌کم ۱۰۰ ساعت کار کرده است. این اطلاعات حذف نمی‌شود و در برازش وارد می‌شود.</p>

<h2 dir="rtl" align="right">چه نتیجه‌ای می‌گیرم؟</h2>

<p dir="rtl" align="right">اگر برآورد ممکن باشد، پارامترهای مدل و اطلاعات محاسبه را دریافت می‌کنید. اگر برآورد ممکن نباشد، علت مشخص می‌شود؛ مثلاً نبود هیچ خرابیِ مشاهده‌شده. مشکل خواندن فایل هم جدا از مشکل آماری گزارش می‌شود. تمام‌شدن محاسبه به‌تنهایی به معنای <strong>مناسب‌بودن مدل</strong><sup id="fnref-model-adequacy"><a href="#fn-model-adequacy">۳</a></sup> برای دادهٔ شما نیست.</p>

<p dir="rtl" align="right">ابزار بررسی مناسب‌بودن مدل فعلاً فقط برای نمونه‌های نماییِ مثبت، متناهی و بدون سانسور در دسترس است. در مسیر انتخاب، از بین گزینه‌هایی که <strong>بررسی کفایت</strong><sup id="fnref-adequacy-check"><a href="#fn-adequacy-check">۴</a></sup> را گذرانده‌اند، گزینهٔ دارای کمترین معیار <bdi dir="ltr">AIC</bdi><sup id="fnref-aic"><a href="#fn-aic">۵</a></sup> انتخاب می‌شود. این معیار هم کیفیت توضیح‌دادن داده و هم تعداد پارامترهای مدل را در نظر می‌گیرد؛ مقدار کمتر، فقط در میان مدل‌های بررسی‌شده، ترجیح دارد. اگر هیچ گزینه‌ای مناسب نباشد، نتیجهٔ <code dir="ltr">NONE_ADEQUATE</code> برمی‌گردد. این قابلیت، رتبه‌بندی خودکار بین نمایی، وایبول و لگ‌نرمال نیست.</p>

<h2 dir="rtl" align="right">به‌جز برازش، چه محاسباتی ممکن است؟</h2>

<p dir="rtl" align="right">برای توزیع‌های <strong>نرمال، گاما، وایبول کمینه، لگ‌نرمال و گامبل راست</strong><sup id="fnref-families"><a href="#fn-families">۶</a></sup> می‌توانید <strong>لگاریتم چگالی</strong><sup id="fnref-log-density"><a href="#fn-log-density">۷</a></sup>، احتمال کمتر یا بیشتر بودن مقدار از یک حد، چندک‌ها<sup id="fnref-quantile"><a href="#fn-quantile">۸</a></sup> و نمونه‌های تصادفی را محاسبه کنید. مثلاً چندک ۹۵٪ مقداری است که طبق مدل، ۹۵٪ داده‌ها پایین‌تر از آن قرار می‌گیرند.</p>

<p dir="rtl" align="right">این ابزارها فعلاً هر بار یک مقدار عددی می‌پذیرند، نه یک آرایه. وجود ابزار محاسبه برای یک توزیع، به معنای وجود ابزار برازش آن نیست. برای نمونه‌گیری نیز مولد اعداد تصادفی را خودتان به برنامه می‌دهید.</p>

<h2 dir="rtl" align="right">اگر داده زیاد باشد یا برنامه قطع شود چه؟</h2>

<p dir="rtl" align="right">در مسیرهای پشتیبانی‌شده می‌توانید داده را بخش‌به‌بخش وارد محاسبه کنید؛ یعنی هر بخش خوانده، به نتیجه افزوده و سپس از حافظه رها می‌شود. به این شیوه <strong>جمع جریانی</strong><sup id="fnref-streaming"><a href="#fn-streaming">۹</a></sup> می‌گوییم. فضای لازم برای نگهداری مجموعِ لگاریتم درست‌نمایی با زیادشدن تعداد داده‌ها رشد نمی‌کند. این ویژگی به‌تنهایی سقف حافظهٔ کل برنامه یا سرعت مشخصی را روی هر کامپیوتر تضمین نمی‌کند.</p>

<p dir="rtl" align="right">شواهد تاریخی این مسیر CSV نمایی روی ۱۰هزار، ۱۰۰هزار و یک‌میلیون ردیف و چند اندازهٔ بخش ثبت شده‌اند. این شواهد نشان می‌دهند که آن آزمایش‌ها کامل شده‌اند، اما تضمین سرعت یا حافظه برای رایانه و دادهٔ شما نیستند.</p>

<p dir="rtl" align="right">برای <strong>محاسبات نماییِ سازگار</strong><sup id="fnref-compatible-exponential"><a href="#fn-compatible-exponential">۱۰</a></sup>، از جمله مسیر CSV مربوط به ادامهٔ اجرا، پیشرفت محاسبه در یک فایل محلی <bdi dir="ltr">SQLite</bdi><sup id="fnref-sqlite"><a href="#fn-sqlite">۱۱</a></sup> ذخیره می‌شود و پس از وقفه قابل ادامه است. پیش از ادامه، برنامه <strong>نسخهٔ منبع</strong><sup id="fnref-source-revision"><a href="#fn-source-revision">۱۲</a></sup>، <strong>جمع کنترلی</strong><sup id="fnref-checksum"><a href="#fn-checksum">۱۳</a></sup>، <strong>نسل وضعیت</strong><sup id="fnref-generation"><a href="#fn-generation">۱۴</a></sup> و <strong>بازه‌های پردازش‌شده</strong><sup id="fnref-ranges"><a href="#fn-ranges">۱۵</a></sup> را بررسی می‌کند. این قابلیت روی یک کامپیوتر و فایل‌سیستم محلی کار می‌کند.</p>

<h2 dir="rtl" align="right">چه کارهایی هنوز پشتیبانی نمی‌شوند؟</h2>

<p dir="rtl" align="right">روش فعلی داده‌هایی را که فقط زمان تقریبی رویداد در یک بازه یا پیش از یک زمان مشخص معلوم است نمی‌پذیرد. مدل‌کردن اثر عواملی مثل دما و فشار بر عمر، اتصال عمومی به دیتافریم و پایگاه داده و ذخیرهٔ پیشرفت بین چند کامپیوتر هم در دسترس نیست. جزئیات فنی این موارد در بخش بازشوندهٔ پایین و فهرست کامل و راست‌چین آن‌ها در <a href="../python/KNOWN_LIMITS.fa.md">محدودیت‌های شناخته‌شده</a> آمده است.</p>

<h2 dir="rtl" align="right">کیفیت کد ما چگونه بررسی می‌شود؟</h2>

<p dir="rtl" align="right">نتایج با محاسبات مرجع مستقل مقایسه می‌شوند. آزمون‌ها دادهٔ نامعتبر، حالت‌های مرزی و قطع و ادامهٔ اجرا را نیز بررسی می‌کنند. این نسخه روی پایتون ۳٫۱۱ تا ۳٫۱۴ آزموده می‌شود.</p>

<p dir="rtl" align="right"><strong>پوشش آزمون</strong><sup id="fnref-coverage"><a href="#fn-coverage">۱۶</a></sup> باید دست‌کم ۹۵٪ باشد؛ یعنی آزمون‌ها باید دست‌کم ۹۵٪ از خط‌های قابل اجرا و ۹۵٪ از مسیرهای تصمیم‌گیریِ کد را اجرا کنند. بخش‌های عددی معیار سخت‌گیرانه‌تری دارند. این عدد شرط پذیرش است، نه گزارش درصد فعلی. در هستهٔ آماری، خطاهایی عمداً وارد کد می‌شوند تا مشخص شود آزمون‌ها آن‌ها را تشخیص می‌دهند یا نه. نتیجهٔ سنجش سرعت و حافظه نیز فقط برای همان داده، محیط و نسخهٔ آزمایش‌شده معتبر است.</p>

<details dir="rtl" align="right">
<summary>جزئیات فنی برای بررسی دقیق‌تر</summary>

<p dir="rtl" align="right">هر سه مدل با <strong>بیشینه‌سازی درست‌نمایی و مکان ثابت صفر</strong><sup id="fnref-mle-location"><a href="#fn-mle-location">۱۷</a></sup> برازش می‌شوند. <strong>در نمایی فقط نرخ، در وایبول شکل و مقیاس و در لگ‌نرمال مکان و مقیاس لگاریتمی</strong><sup id="fnref-parameters"><a href="#fn-parameters">۱۸</a></sup> برآورد می‌شود. <strong>وایبول و لگ‌نرمال وزن فراوانی می‌پذیرند</strong><sup id="fnref-frequency-weights"><a href="#fn-frequency-weights">۱۹</a></sup>؛ یعنی وزن، تعداد تکرار مشاهده را نشان می‌دهد. در وایبول می‌توان شکل را از پیش ثابت کرد. <strong>شکست عددی</strong><sup id="fnref-numerical-failure"><a href="#fn-numerical-failure">۲۰</a></sup> یا نرسیدن روش برازش به جواب، با علت مشخص گزارش می‌شود.</p>

<p dir="rtl" align="right">ارزیابی نمایی شامل آزمون‌های <bdi dir="ltr">KS/AD/CvM</bdi><sup id="fnref-gof-tests"><a href="#fn-gof-tests">۲۱</a></sup> با شبیه‌سازی مونت‌کارلو و برازش مجدد، معیارهای <bdi dir="ltr">AIC/BIC</bdi><sup id="fnref-bic"><a href="#fn-bic">۲۲</a></sup> و خلاصهٔ کالیبراسیون است. شمار برازش‌های مجدد درخواستی، موفق و ناموفق و عدم‌قطعیت مونت‌کارلو گزارش می‌شود. برای اینکه آزمایش قابل تکرار باشد، شما دنبالهٔ اعداد تصادفی را انتخاب می‌کنید؛ مثلاً با یک <bdi dir="ltr">seed</bdi><sup id="fnref-seed"><a href="#fn-seed">۲۳</a></sup> مشخص می‌توانید همان آزمایش را دوباره اجرا کنید.</p>

<p dir="rtl" align="right">در جمع جریانی، عددهای حاصل از هر بخش با <strong>قالب عدد اعشاری رایانه</strong><sup id="fnref-binary64"><a href="#fn-binary64">۲۴</a></sup>، یعنی <bdi dir="ltr">binary64</bdi>، جمع می‌شوند و مجموع فقط یک بار در پایان گرد می‌شود. شمار مشاهدات <strong>سقف صریح عدد صحیح</strong><sup id="fnref-unsigned-limit"><a href="#fn-unsigned-limit">۲۵</a></sup> بدون علامت ۶۴بیتی دارد. آزمون‌ها وقفه، اجرای مجدد، خرابی اطلاعات، رقابت دسترسی و لغو را پوشش می‌دهند.</p>

<p dir="rtl" align="right"><strong>سانسور چپ و فاصله‌ای</strong><sup id="fnref-left-interval"><a href="#fn-left-interval">۲۶</a></sup>، <strong>برش داده</strong><sup id="fnref-truncation"><a href="#fn-truncation">۲۷</a></sup>، <strong>متغیرهای کمکی</strong><sup id="fnref-covariates"><a href="#fn-covariates">۲۸</a></sup>، <strong>وزن تحلیلی</strong><sup id="fnref-analytic-weights"><a href="#fn-analytic-weights">۲۹</a></sup>، <strong>پارامتر مکان آزاد</strong><sup id="fnref-free-location"><a href="#fn-free-location">۳۰</a></sup>، API آرایه‌ای، ذخیرهٔ وضعیت توزیع‌شده، آداپتور عمومی دیتافریم یا پایگاه داده، پایداری انتخاب با <strong>بوت‌استرپ</strong><sup id="fnref-bootstrap"><a href="#fn-bootstrap">۳۱</a></sup> و استنباط برای تمام خانواده‌ها پشتیبانی نمی‌شوند. شواهد انتشار و مقیاس به شناسهٔ کامل کامیت متصل‌اند. نمونهٔ CSV و گزارش‌های فارسی نیز در آزمون‌های خودکار بررسی می‌شوند.</p>

</details>

<h2 dir="rtl" align="right">پانویس اصطلاحات</h2>

<p id="fn-fitting" dir="rtl" align="right"><strong>۱.</strong> <bdi dir="ltr">Distribution Fitting</bdi> — برآورد پارامترهای یک مدل توزیع از روی داده. <a href="#fnref-fitting">↩</a></p>
<p id="fn-censoring" dir="rtl" align="right"><strong>۲.</strong> <bdi dir="ltr">Right Censoring</bdi> — رویداد تا پایان مشاهده دیده نشده و زمان نهایی آن معلوم نیست. <a href="#fnref-censoring">↩</a></p>
<p id="fn-aic" dir="rtl" align="right"><strong>۵.</strong> <bdi dir="ltr">Akaike Information Criterion (AIC)</bdi> — معیار مقایسهٔ مدل‌ها که کیفیت برازش و تعداد پارامترها را با هم می‌سنجد. مقدار کمتر فقط در میان مدل‌های بررسی‌شده ترجیح دارد. <a href="#fnref-aic">↩</a></p>
<p id="fn-quantile" dir="rtl" align="right"><strong>۸.</strong> <bdi dir="ltr">Quantile</bdi> — آستانه‌ای که سهم معینی از داده‌ها یا احتمال مدل پایین‌تر از آن قرار می‌گیرد. <a href="#fnref-quantile">↩</a></p>
<p id="fn-streaming" dir="rtl" align="right"><strong>۹.</strong> <bdi dir="ltr">Streaming reduction</bdi> — پردازش پیاپی بخش‌های داده، بدون نگه‌داشتن تمام داده در حافظه. <a href="#fnref-streaming">↩</a></p>
<p id="fn-sqlite" dir="rtl" align="right"><strong>۱۱.</strong> <bdi dir="ltr">SQLite</bdi> — پایگاه دادهٔ کوچکِ فایل‌محور که در همان رایانه نگه‌داری می‌شود. <a href="#fnref-sqlite">↩</a></p>
<p id="fn-source-revision" dir="rtl" align="right"><strong>۱۲.</strong> <bdi dir="ltr">Source revision</bdi> — شناسه‌ای که نشان می‌دهد دادهٔ ورودی از زمان ذخیرهٔ وضعیت تغییر نکرده است. <a href="#fnref-source-revision">↩</a></p>
<p id="fn-checksum" dir="rtl" align="right"><strong>۱۳.</strong> <bdi dir="ltr">Checksum</bdi> — عددی محاسبه‌شده از اطلاعات ذخیره‌شده که تغییر یا خرابی ناخواستهٔ آن را آشکار می‌کند. <a href="#fnref-checksum">↩</a></p>
<p id="fn-generation" dir="rtl" align="right"><strong>۱۴.</strong> <bdi dir="ltr">Checkpoint generation</bdi> — شمارهٔ نسخهٔ متوالیِ وضعیت ذخیره‌شده؛ برای جلوگیری از ثبت هم‌زمان و ناسازگار دو ادامهٔ اجرا به کار می‌رود. <a href="#fnref-generation">↩</a></p>
<p id="fn-ranges" dir="rtl" align="right"><strong>۱۵.</strong> <bdi dir="ltr">Processed ranges</bdi> — بخش‌هایی از ترتیب داده که پیش‌تر با موفقیت محاسبه شده‌اند. <a href="#fnref-ranges">↩</a></p>
<p id="fn-coverage" dir="rtl" align="right"><strong>۱۶.</strong> <bdi dir="ltr">Test coverage</bdi> — نسبت خط‌ها و مسیرهای تصمیم‌گیریِ کد که هنگام اجرای آزمون‌ها واقعاً اجرا می‌شوند. <a href="#fnref-coverage">↩</a></p>
<p id="fn-binary64" dir="rtl" align="right"><strong>۲۴.</strong> <bdi dir="ltr">binary64</bdi> — قالب معمول ۶۴بیتی برای نگه‌داری عددهای اعشاری در رایانه. <a href="#fnref-binary64">↩</a></p>
<p id="fn-model-adequacy" dir="rtl" align="right"><strong>۳.</strong> <bdi dir="ltr">Model adequacy</bdi> — اینکه فرض‌ها و شکل مدل برای داده و هدف تحلیل شما پذیرفتنی باشند. <a href="#fnref-model-adequacy">↩</a></p>
<p id="fn-adequacy-check" dir="rtl" align="right"><strong>۴.</strong> <bdi dir="ltr">Adequacy check</bdi> — بررسی آماریِ تعریف‌شده در این مسیر برای رد یا نپذیرفتن یک مدل نامزد. <a href="#fnref-adequacy-check">↩</a></p>
<p id="fn-families" dir="rtl" align="right"><strong>۶.</strong> خانواده‌های توزیع احتمال با شکل‌ها و کاربردهای متفاوت؛ این فهرست فقط عملیات اسکالر دارد. <a href="#fnref-families">↩</a></p>
<p id="fn-log-density" dir="rtl" align="right"><strong>۷.</strong> <bdi dir="ltr">Log density</bdi> — لگاریتم احتمال‌پذیری نسبی یک مشاهده زیر مدل؛ برای محاسبات پایدار عددی به کار می‌رود. <a href="#fnref-log-density">↩</a></p>
<p id="fn-compatible-exponential" dir="rtl" align="right"><strong>۱۰.</strong> اجرای نمایی با قرارداد یکسانِ منبع، مدل و روش تجمیع؛ فقط چنین اجرایی می‌تواند از وضعیت ذخیره‌شده ادامه یابد. <a href="#fnref-compatible-exponential">↩</a></p>
<p id="fn-mle-location" dir="rtl" align="right"><strong>۱۷.</strong> <bdi dir="ltr">Maximum likelihood estimation</bdi> پارامترهایی را انتخاب می‌کند که دادهٔ مشاهده‌شده را زیر مدل محتمل‌تر می‌کنند. مکان ثابت صفر یعنی مدل اجازهٔ جابه‌جایی افقیِ توزیع را نمی‌دهد. <a href="#fnref-mle-location">↩</a></p>
<p id="fn-parameters" dir="rtl" align="right"><strong>۱۸.</strong> نرخ در نمایی سرعت رخداد است؛ شکل و مقیاس در وایبول روند و واحد زمانی را کنترل می‌کنند؛ پارامترهای لگ‌نرمال روی لگاریتم زمان کار می‌کنند. <a href="#fnref-parameters">↩</a></p>
<p id="fn-frequency-weights" dir="rtl" align="right"><strong>۱۹.</strong> <bdi dir="ltr">Frequency weights</bdi> یعنی هر مشاهده به تعداد مشخصی تکرار شده است؛ این با وزن تحلیلی متفاوت است. <a href="#fnref-frequency-weights">↩</a></p>
<p id="fn-numerical-failure" dir="rtl" align="right"><strong>۲۰.</strong> <bdi dir="ltr">Numerical failure</bdi> — محدودیت محاسبات اعشاری یا همگرایی باعث می‌شود نتیجهٔ قابل‌اعتماد تولید نشود. <a href="#fnref-numerical-failure">↩</a></p>
<p id="fn-gof-tests" dir="rtl" align="right"><strong>۲۱.</strong> <bdi dir="ltr">KS/AD/CvM</bdi> — سه آزمون برای سنجش فاصلهٔ داده از توزیع مدل‌شده؛ هرکدام به بخش متفاوتی از تفاوت‌ها حساس‌اند. <a href="#fnref-gof-tests">↩</a></p>
<p id="fn-bic" dir="rtl" align="right"><strong>۲۲.</strong> <bdi dir="ltr">BIC</bdi> — معیار مقایسهٔ مدل‌ها که برای مدل‌های دارای پارامترهای بیشتر جریمهٔ بیشتری در نظر می‌گیرد. <a href="#fnref-bic">↩</a></p>
<p id="fn-seed" dir="rtl" align="right"><strong>۲۳.</strong> <bdi dir="ltr">Seed</bdi> — مقدار آغازینِ تولید اعداد تصادفی که تکرار همان دنباله را ممکن می‌کند. <a href="#fnref-seed">↩</a></p>
<p id="fn-unsigned-limit" dir="rtl" align="right"><strong>۲۵.</strong> بیشترین تعداد مشاهده‌ای که این قرارداد می‌پذیرد؛ از عدد منفی استفاده نمی‌شود و حد آن با ۶۴ بیت مشخص شده است. <a href="#fnref-unsigned-limit">↩</a></p>
<p id="fn-left-interval" dir="rtl" align="right"><strong>۲۶.</strong> سانسور چپ یعنی رویداد پیش از یک زمان رخ داده، و سانسور فاصله‌ای یعنی رویداد در بازه‌ای رخ داده است، بدون زمان دقیق. <a href="#fnref-left-interval">↩</a></p>
<p id="fn-truncation" dir="rtl" align="right"><strong>۲۷.</strong> <bdi dir="ltr">Truncation</bdi> — ورود مشاهده به داده به عبور از یک شرط یا حد وابسته بوده است. <a href="#fnref-truncation">↩</a></p>
<p id="fn-covariates" dir="rtl" align="right"><strong>۲۸.</strong> <bdi dir="ltr">Covariates</bdi> — متغیرهایی مانند دما یا فشار که ممکن است با عمر ارتباط داشته باشند. <a href="#fnref-covariates">↩</a></p>
<p id="fn-analytic-weights" dir="rtl" align="right"><strong>۲۹.</strong> وزن‌هایی که اهمیت یا دقت مشاهده‌ها را تغییر می‌دهند، نه تعداد تکرار آن‌ها. <a href="#fnref-analytic-weights">↩</a></p>
<p id="fn-free-location" dir="rtl" align="right"><strong>۳۰.</strong> پارامتر مکان قابل برآورد که توزیع را روی محور زمان جابه‌جا می‌کند. <a href="#fnref-free-location">↩</a></p>
<p id="fn-bootstrap" dir="rtl" align="right"><strong>۳۱.</strong> <bdi dir="ltr">Bootstrap</bdi> — نمونه‌گیری تکراری از داده برای سنجش پایداری یک نتیجه. <a href="#fnref-bootstrap">↩</a></p>

<p dir="rtl" align="right"><a href="../README.fa.md">بازگشت به راهنمای اصلی</a></p>
