<h1 dir="rtl" align="right">محدودیت‌های شناخته‌شدهٔ <bdi dir="ltr">Veridist</bdi> ۱٫۰</h1>

<p dir="rtl" align="right"><a href="KNOWN_LIMITS.md">English</a> | <a href="KNOWN_LIMITS.fa.md">فارسی</a> | <a href="KNOWN_LIMITS.de.md">Deutsch</a></p>

<!-- CI contract IDs: `FIT-CSV-EXP` `CSV-STRICT` `SCALAR-FAMILIES` `STREAM-SOURCE` `MEMORY-BOUND` `SCALE-EVIDENCE` `LICENSE` -->

<p dir="rtl" align="right">این صفحه مرزهای نسخهٔ <bdi dir="ltr">1.0.1</bdi> را به زبان ساده توضیح می‌دهد. اگر کاری در این فهرست نیامده، نباید آن را قابلیت پشتیبانی‌شده فرض کرد.</p>

<h2 dir="rtl" align="right">این محدودیت‌ها در کاربردهای رایج چه معنایی دارند؟</h2>

<ul dir="rtl" align="right">
  <li>تیم‌های قابلیت اطمینان، سلامت، اعتبار، بیمه، محصول دیجیتال و عملیات می‌توانند یک «زمان تا رویداد» روشن را، در صورت سازگاری با فرض‌های مستند، مدل کنند. مدل‌های فعلی اثر ویژگی‌های مشتری، بیمار، دستگاه یا محیط را بر آن رویداد تعدیل نمی‌کنند.</li>
  <li>تیم‌های تقلب و امنیت سایبری می‌توانند با محاسبات اسکالرِ پشتیبانی‌شده و یک مدل مرجع که از قبل مشخص شده، یک نشانه بسازند. <bdi dir="ltr">Veridist</bdi> طبقه‌بند آموزش نمی‌دهد، آستانهٔ هشدار انتخاب نمی‌کند، برچسب‌های بازخورد را پردازش نمی‌کند و آداپتور آمادهٔ جریان رخدادِ محیط عملیاتی ندارد.</li>
  <li>تیم‌های مالی، بیمه، تولید و زنجیرهٔ تأمین نباید فرض کنند همهٔ خانواده‌های ثبت‌شده API برازش دارند. محاسبات اسکالر به خانواده و پارامترهایی نیاز دارند که جداگانه توجیه شده باشند، مگر اینکه مسیر برازش مستندی برای آن‌ها وجود داشته باشد.</li>
  <li>در هر حوزه، خروجی مدل همچنان به اعتبارسنجی تخصصی، نمونه‌گیری مناسب، بررسی هزینهٔ تصمیم و ارزیابی‌های قانونی، بالینی، ایمنی یا نظارتیِ لازم نیاز دارد.</li>
</ul>

<h2 dir="rtl" align="right">ورود داده از فایل</h2>

<ul dir="rtl" align="right">
  <li><bdi dir="ltr">CSV</bdi> فقط برای مدل نمایی با <strong>مکان ثابت صفر و پارامتر نرخ</strong><sup id="fnref-location-rate"><a href="#fn-location-rate">۱</a></sup> است.</li>
  <li>فایل باید <bdi dir="ltr">UTF-8</bdi> باشد و دقیقاً دو ستون <code dir="ltr">time,event_observed</code> داشته باشد.</li>
  <li>عدد ۱ در ستون وضعیت یعنی خرابی دیده شده و عدد ۰ یعنی دستگاه تا پایان مشاهده هنوز خراب نشده است.</li>
  <li><strong>وایبول کمینه و لگ‌نرمال</strong><sup id="fnref-families"><a href="#fn-families">۲</a></sup> با دادهٔ آماده‌شده در پایتون کار می‌کنند، نه با یک API عمومی برای فایل.</li>
</ul>

<h2 dir="rtl" align="right">چیزهایی که فعلاً پشتیبانی نمی‌شوند</h2>

<ul dir="rtl" align="right">
  <li><strong>سانسور چپ، سانسور فاصله‌ای و دادهٔ برش‌خورده</strong><sup id="fnref-censoring-truncation"><a href="#fn-censoring-truncation">۳</a></sup></li>
  <li><strong>متغیرهای کمکی</strong><sup id="fnref-covariates"><a href="#fn-covariates">۴</a></sup> مانند دما و فشار</li>
  <li><strong>وزن تحلیلی و پارامتر مکان آزاد</strong><sup id="fnref-weights-location"><a href="#fn-weights-location">۵</a></sup></li>
  <li>ورودی آرایه‌ای، آداپتور آماده برای Parquet، Arrow، دیتافریم، پایگاه داده یا شبکه</li>
  <li>ذخیرهٔ وضعیت میان چند رایانه</li>
  <li>استنباط آماری برای همهٔ توزیع‌ها و سنجش پایداری انتخاب مدل با <strong>بوت‌استرپ</strong><sup id="fnref-bootstrap"><a href="#fn-bootstrap">۶</a></sup></li>
</ul>

<h2 dir="rtl" align="right">دادهٔ بزرگ و ادامهٔ اجرا</h2>

<ul dir="rtl" align="right">
  <li>پردازش مرحله‌ای داده به معنی تضمین سرعت یا سقف حافظه برای همهٔ رایانه‌ها نیست.</li>
  <li>ادعای مقیاس فقط برای همان آداپتور، مدل، داده، محیط، نسخهٔ پایتون و کامیتی معتبر است که آزمایش شده‌اند.</li>
  <li>ادامهٔ اجرا فقط برای مسیر CSV طول عمر و <bdi dir="ltr">SQLite</bdi><sup id="fnref-sqlite"><a href="#fn-sqlite">۷</a></sup> محلی پشتیبانی می‌شود.</li>
</ul>

<h2 dir="rtl" align="right">مجوز</h2>

<p dir="rtl" align="right">بسته با <bdi dir="ltr">BUSL-1.1</bdi> منتشر می‌شود. شرایط استفادهٔ اضافی با <bdi dir="ltr">Apache-2.0</bdi> و تاریخ تغییر مجوز در <a href="../LICENSE"><bdi dir="ltr">LICENSE</bdi></a> آمده است؛ تاریخ تغییر مجوز <bdi dir="ltr">2030-09-05</bdi> است.</p>

<h2 dir="rtl" align="right">پانویس اصطلاحات</h2>

<p id="fn-location-rate" dir="rtl" align="right"><strong>۱.</strong> مکان ثابت صفر یعنی توزیع روی محور زمان جابه‌جا نمی‌شود. نرخ، سرعت رخداد در مدل نمایی است. <a href="#fnref-location-rate">↩</a></p>
<p id="fn-families" dir="rtl" align="right"><strong>۲.</strong> وایبول کمینه می‌تواند نرخ خرابیِ کاهشی، ثابت یا افزایشی را توصیف کند؛ لگ‌نرمال برای زمان‌های مثبت با ساختار لگاریتمی به کار می‌رود. <a href="#fnref-families">↩</a></p>
<p id="fn-censoring-truncation" dir="rtl" align="right"><strong>۳.</strong> در سانسور چپ فقط می‌دانیم رویداد پیش از زمانی رخ داده است؛ در سانسور فاصله‌ای می‌دانیم در یک بازه رخ داده؛ در برش داده، ورود مشاهده به داده به یک شرط وابسته بوده است. <a href="#fnref-censoring-truncation">↩</a></p>
<p id="fn-covariates" dir="rtl" align="right"><strong>۴.</strong> متغیرهای کمکی اطلاعاتی مانند دما یا فشارند که ممکن است با عمر ارتباط داشته باشند. <a href="#fnref-covariates">↩</a></p>
<p id="fn-weights-location" dir="rtl" align="right"><strong>۵.</strong> وزن تحلیلی اهمیت یا دقت مشاهده را تغییر می‌دهد؛ پارامتر مکان آزاد توزیع را روی محور زمان جابه‌جا می‌کند. <a href="#fnref-weights-location">↩</a></p>
<p id="fn-bootstrap" dir="rtl" align="right"><strong>۶.</strong> بوت‌استرپ نمونه‌گیری تکراری از داده برای سنجش پایداری نتیجه است. <a href="#fnref-bootstrap">↩</a></p>
<p id="fn-sqlite" dir="rtl" align="right"><strong>۷.</strong> SQLite پایگاه دادهٔ کوچک و فایل‌محور روی همان رایانه است؛ وضعیت اجرا در آن ذخیره می‌شود تا اجرای سازگار پس از وقفه ادامه یابد. <a href="#fnref-sqlite">↩</a></p>

<p dir="rtl" align="right"><a href="../README.fa.md">بازگشت به راهنمای اصلی</a> | <a href="../docs/capability-guide.fa.md">راهنمای قابلیت‌ها</a></p>
