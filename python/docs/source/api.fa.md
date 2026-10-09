<a id="veridist-api-fa"></a>
<h1 dir="rtl" align="right">راهنمای API در <bdi dir="ltr">Veridist</bdi></h1>

<p dir="rtl" align="right"><a href="api.md">انگلیسی</a> | <a href="api.fa.md">فارسی</a> | <a href="api.de.md">آلمانی</a></p>

<p dir="rtl" align="right">این راهنما API عمومیِ فعلی <bdi dir="ltr">Veridist</bdi> 2.0 را توضیح می‌دهد. برای برازش توزیع<sup id="fnref-fitting"><a href="#fn-fitting">۱</a></sup> روی داده‌های طول عمر یا اندازه‌گیری، معمولاً مشاهدات خود را، یا یک فایل CSV را، به یکی از توابع برازش می‌دهید و نتیجهٔ برگردانده‌شده را می‌خوانید. شش خانوادهٔ توزیع برازش دارند، هر عملیات توزیعی آرایه‌های numpy را نیز می‌پذیرد و هر برازش موفق می‌تواند عدم‌قطعیتِ برآورد خود را گزارش کند<sup id="fnref-uncertainty"><a href="#fn-uncertainty">۲۲</a></sup>. اگر محاسبه طولانی است و ممکن است متوقف شود، می‌توانید پیشرفت آن را ذخیره کنید و بعداً ادامه دهید. ابزارهای اسکالر<sup id="fnref-scalar"><a href="#fn-scalar">۲</a></sup> و جریان‌های داده‌ای که مدیریتشان با فراخواننده<sup id="fnref-caller"><a href="#fn-caller">۳</a></sup> است نیز برای استفاده‌های فنی‌تر در دسترس‌اند. اگر از نسخهٔ ۱٫۰ مهاجرت می‌کنید، <a href="../migration-2.0.md">راهنمای مهاجرت</a> را بخوانید.</p>

<h2 dir="rtl" align="right">از همین قرارداد زمان و رویداد در حوزه‌های مختلف استفاده کنید</h2>

<p dir="rtl" align="right">دو ستون فایل <bdi dir="ltr">CSV</bdi> یک مشاهدهٔ عمومیِ «زمان تا رویداد» را توصیف می‌کنند. ستون <code dir="ltr">time</code> می‌گوید هر مورد چه مدت مشاهده شده و ستون <code dir="ltr">event_observed</code> مشخص می‌کند رویداد انتخاب‌شده در آن مدت رخ داده است یا نه. کد معنای رویداد را تعیین نمی‌کند؛ شما باید آن را در کل تحلیل یکسان و روشن تعریف کنید.</p>

<div dir="rtl" align="right">

| حوزهٔ کاربرد | معنای احتمالی <code dir="ltr">time</code> | معنای احتمالی رویداد <code dir="ltr">1</code> | معنای احتمالی رویداد <code dir="ltr">0</code> |
| --- | --- | --- | --- |
| سابقهٔ قابلیت اطمینان | ساعت‌های مشاهدهٔ یک قطعه | قطعه خراب شده است | قطعه هنگام پایان مشاهده هنوز کار می‌کرد |
| سابقهٔ پژوهش سلامت | روزهای پیگیری یک فرد | رخداد سلامتِ تعریف‌شده اتفاق افتاده است | تا آخرین پیگیری رخداد اتفاق نیفتاده است |
| سابقهٔ اعتبار یا بیمه | ماه‌های پیگیری حساب یا بیمه‌نامه | نکول یا نخستین خسارت رخ داده است | تا پایان بررسی چنین رخدادی ثبت نشده است |
| سابقهٔ محصول دیجیتال | روزهای گذشته از ثبت‌نام یا ورود به کمپین | ریزش یا تبدیل رخ داده است | کاربر تا پایان بازه فعال مانده و رویداد رخ نداده است |
| سابقهٔ عملیات | ساعت‌های گذشته از شروع تعمیر، سفارش یا پرونده | فرایند کامل شده است | هنگام پایان جمع‌آوری داده، فرایند هنوز باز بوده است |

</div>

<p dir="rtl" align="right">در کشف تقلب یا امنیت سایبری، فایل زمان تا رویداد می‌تواند برای پرسشی مانند «زمان تا نخستین هشدار» مناسب باشد، اما قالب عمومی داده‌های تقلب نیست. ابزارهای توزیعیِ سطح پایین می‌توانند مبلغ تراکنش، فاصلهٔ زمانی یا زمان پاسخ را با یک توزیع مرجع که پارامترهایش از قبل مشخص شده مقایسه کنند. نتیجهٔ این محاسبه فقط یکی از ورودی‌های یک سامانهٔ تشخیص جداگانه و آزموده‌شده است و به‌تنهایی تصمیم تقلب نیست.</p>

<h2 dir="rtl" align="right">مسیر اجرا را انتخاب کنید</h2>

<table dir="rtl" width="100%">
  <thead>
    <tr>
      <th width="36%" align="center"><p align="center">کاری که می‌خواهید انجام دهید</p></th>
      <th width="32%" align="center"><p align="center">تابع مربوط</p></th>
      <th width="32%" align="center"><p align="center">محدودیت این روش</p></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="right">خواندن فایل CSV و برازش مدل نمایی در یک اجرا</td>
      <td dir="ltr" align="left"><code class="literal">fit_exponential_csv</code></td>
      <td align="right">امکان توقف و ادامه، لغو اجرا یا انتخاب خودکار مدل را ندارد.</td>
    </tr>
    <tr>
      <td align="right">ادامهٔ محاسبه با داده‌های JSON بخش‌بندی‌شده</td>
      <td dir="ltr" align="left"><code class="literal">fit_exponential_checkpointed_chunks</code></td>
      <td align="right">آماده‌کردن و خواندن بخش‌های داده با برنامهٔ شماست.</td>
    </tr>
    <tr>
      <td align="right">پردازش فایل CSV با امکان لغو و ادامه از آخرین سطر ذخیره‌شده</td>
      <td dir="ltr" align="left"><code class="literal">fit_exponential_checkpointed_csv</code></td>
      <td align="right">فقط برای فایل محلی روی یک رایانه است؛ بازیابی توزیع‌شده<sup id="fnref-distributed-recovery"><a href="#fn-distributed-recovery">۴</a></sup> ندارد.</td>
    </tr>
    <tr>
      <td align="right">برازش یکی از شش خانواده روی مشاهداتی که در حافظه‌اند</td>
      <td dir="ltr" align="left"><code class="literal">fit</code>، <code class="literal">fit_weibull</code>، <code class="literal">fit_gamma</code> و سایر برازش‌های هر خانواده</td>
      <td align="right">همهٔ مشاهدات باید در حافظه جا شوند؛ انتخاب خودکار مدل وجود ندارد.</td>
    </tr>
    <tr>
      <td align="right">محاسبهٔ چگالی، CDF، بقا، چندک یا نمونه برای یک مقدار یا یک آرایه</td>
      <td dir="ltr" align="left"><code class="literal">logpdf</code>، <code class="literal">cdf</code>، <code class="literal">sf</code>، <code class="literal">ppf</code>، <code class="literal">sample</code></td>
      <td align="right">از توزیعی استفاده می‌کنند که از قبل مشخص کرده‌اید؛ پارامتر برازش نمی‌کنند.</td>
    </tr>
    <tr>
      <td align="right">محاسبهٔ چگالی لگاریتمی برای یک مقدار</td>
      <td dir="ltr" align="left"><code class="literal">evaluate_log_density</code></td>
      <td align="right">پارامترها را برازش نمی‌کند.</td>
    </tr>
    <tr>
      <td align="right">محاسبهٔ درست‌نمایی برای چند بخش داده، با سانسور راست یا بدون آن</td>
      <td dir="ltr" align="left"><code class="literal">reduce_log_likelihood_chunks</code>، <code class="literal">reduce_lifetime_log_likelihood_chunks</code>، <code class="literal">reduce_value_log_likelihood_chunks</code></td>
      <td align="right">بهترین توزیع را انتخاب نمی‌کنند.</td>
    </tr>
  </tbody>
</table>

<p dir="rtl" align="right">برای حالت معمول، نمونهٔ کامل زیر را اجرا کنید:</p>

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from veridist import (
    CsvLifetimeLimits,
    CsvLifetimeSchema,
    PublicSourceId,
    fit_exponential_csv,
)
from veridist.families import ExponentialFitSuccess

with TemporaryDirectory() as directory:
    path = Path(directory) / "lifetimes.csv"
    path.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
    result = fit_exponential_csv(
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32768, 32768),
    )

fit = result.fit
assert isinstance(fit, ExponentialFitSuccess)
print(f"rate={fit.rate}; events={fit.event_count}; censored={fit.censored_count}")
```

```text
rate=0.5; events=1; censored=1
```

<p dir="rtl" align="right">در این مثال دو مشاهده داریم. در مشاهدهٔ اول، رویداد در زمان ۱ رخ داده است. در مشاهدهٔ دوم، تا زمان ۱ رویدادی ندیده‌ایم و فقط می‌دانیم عمر واقعی از ۱ بیشتر است. بنابراین یک رخداد ثبت‌شده و مجموعاً ۲ واحد زمانِ تحت مشاهده داریم. نرخ<sup id="fnref-rate"><a href="#fn-rate">۵</a></sup> برآوردشده برابر <code dir="ltr">1 / 2 = 0.5</code> رخداد در هر واحد زمان است. یعنی مدل برآورد می‌کند که در واحدهای مشابه، به‌طور میانگین به ازای هر واحد زمان مشاهده‌شده نیم رخداد روی می‌دهد. برای محاسبهٔ احتمال رخداد، باید بازهٔ زمانی مشخصی هم تعیین شود.</p>

<h2 dir="rtl" align="right">فایل CSV باید چه شکلی باشد؟</h2>

<p dir="rtl" align="right">تابع <code dir="ltr">fit_exponential_csv(path, *, schema, source_id, limits)</code> یک فایل UTF-8 با دو ستون <code dir="ltr">time,event_observed</code> و دقیقاً همین ترتیب می‌پذیرد. مقدار <code dir="ltr">time</code> باید عددی متناهی و نامنفی باشد. در ستون <code dir="ltr">event_observed</code> مقدار <code dir="ltr">1</code> یعنی رویداد در زمان ثبت‌شده رخ داده است؛ مقدار <code dir="ltr">0</code> یعنی تا پایان زمان ثبت‌شده هنوز رویداد را ندیده‌ایم. حالت دوم سانسور راست مستقل<sup id="fnref-right-censoring"><a href="#fn-right-censoring">۶</a></sup> نام دارد.</p>

<p dir="rtl" align="right"><code dir="ltr">CsvLifetimeSchema</code> نام دو ستون مورد انتظار را مشخص می‌کند. <code dir="ltr">PublicSourceId</code> یک شناسهٔ عمومی و غیرمحرمانه برای ثبت منشأ<sup id="fnref-provenance"><a href="#fn-provenance">۷</a></sup> داده است؛ مسیر محلی فایل در گزارش نتیجه قرار نمی‌گیرد. <code dir="ltr">CsvLifetimeLimits</code> سقف اندازهٔ هر بخش از داده و حداکثر حجم داده‌ای را که هم‌زمان در صف پردازش نگه داشته می‌شود تعیین می‌کند<sup id="fnref-byte-limits"><a href="#fn-byte-limits">۸</a></sup>. هر دو مقدار باید مثبت باشند.</p>

<p dir="rtl" align="right"><bdi dir="ltr">Veridist</bdi> نام ستون‌ها، جداکننده، کدگذاری، دادهٔ گمشده یا معنی صفر و یک را حدس نمی‌زند. اگر فایل با قالب بالا سازگار نباشد، پردازش متوقف می‌شود و خطا به‌روشنی گزارش می‌شود؛ برنامه مقدارهای مشکوک را خودکار تغییر نمی‌دهد.</p>

<h2 dir="rtl" align="right">نتیجه را چگونه بخوانید؟</h2>

<p dir="rtl" align="right">تابع <code dir="ltr">fit_exponential_csv</code> همیشه یک شیء از نوع <code dir="ltr">ExponentialSourceFitResult</code> برمی‌گرداند. اگر فایل با موفقیت پردازش شود و داده برای برآورد نرخ کافی باشد، نتیجهٔ برازش در <code dir="ltr">result.fit</code> قرار می‌گیرد. اگر فایل درست پردازش شود ولی از نظر آماری نتوان نرخ معتبری برآورد کرد—برای مثال فایل خالی باشد یا هیچ رخدادی مشاهده نشده باشد—همان فیلد یک عدم‌برآورد آماری نوع‌دار<sup id="fnref-typed-non-estimate"><a href="#fn-typed-non-estimate">۹</a></sup> برمی‌گرداند تا دلیل مشخص بماند.</p>

<p dir="rtl" align="right">اگر خواندن یا پردازش فایل شکست بخورد، <code dir="ltr">result.fit</code> برابر <code dir="ltr">None</code> است. در این حالت <code dir="ltr">result.execution</code> یک خروجی نوع‌دار<sup id="fnref-typed-outcome"><a href="#fn-typed-outcome">۱۰</a></sup> شامل مرحله و دلیل شکست دارد. پیش از استفاده از پارامترهای مدل، ابتدا نوع نتیجه را بررسی کنید.</p>

<p dir="rtl" align="right">مدل CSV پارامتر مکان را روی صفر ثابت نگه می‌دارد. برازشی که برمی‌گرداند می‌تواند عدم‌قطعیت برآورد خود را گزارش کند (بخش عدم‌قطعیت را در ادامه ببینید)، اما این مسیر آزمون مناسب‌بودن مدل<sup id="fnref-goodness-of-fit"><a href="#fn-goodness-of-fit">۱۲</a></sup>، وزن<sup id="fnref-weight"><a href="#fn-weight">۱۳</a></sup>، متغیر کمکی<sup id="fnref-covariate"><a href="#fn-covariate">۱۴</a></sup>، برش داده<sup id="fnref-truncation"><a href="#fn-truncation">۱۵</a></sup>، سانسور چپ<sup id="fnref-left-censoring"><a href="#fn-left-censoring">۱۶</a></sup>، سانسور فاصله‌ای<sup id="fnref-interval-censoring"><a href="#fn-interval-censoring">۱۷</a></sup>، پارامتر مکان آزاد<sup id="fnref-free-location"><a href="#fn-free-location">۱۸</a></sup> یا انتخاب خودکار مدل را ارائه نمی‌کند. جزئیات فرض آماری و حالت‌های ناموفق در <a href="exponential-right-censoring.md">آموزش داده‌های سانسورشده از راست</a> آمده است.</p>

<h2 dir="rtl" align="right">برازش هر یک از شش خانواده</h2>

<p dir="rtl" align="right"><code dir="ltr">fit(family, observations)</code> یکی از شش خانواده را با بیشینه‌سازی درست‌نمایی روی مشاهداتی که در حافظه نگه داشته شده‌اند برازش می‌دهد. <code dir="ltr">family</code> یک <code dir="ltr">FamilyId</code> یا مقدار رشته‌ای آن است: <code dir="ltr">exponential</code>، <code dir="ltr">weibull_min</code>، <code dir="ltr">lognormal</code>، <code dir="ltr">gamma</code>، <code dir="ltr">normal</code> یا <code dir="ltr">gumbel_right</code>. هر خانواده تابع مخصوص خود را هم دارد (<code dir="ltr">fit_exponential</code>، <code dir="ltr">fit_weibull</code>، <code dir="ltr">fit_lognormal</code>، <code dir="ltr">fit_gamma</code>، <code dir="ltr">fit_normal</code> و <code dir="ltr">fit_gumbel_right</code>) که همان مشاهدات و گزینه‌ها را می‌گیرد.</p>

<p dir="rtl" align="right">چهار خانوادهٔ طول عمر (<code dir="ltr">exponential</code>، <code dir="ltr">weibull_min</code>، <code dir="ltr">lognormal</code> و <code dir="ltr">gamma</code>) مشاهدات <code dir="ltr">ExactLifetime</code> و <code dir="ltr">RightCensoredLifetime</code> را می‌گیرند. دو خانواده‌ای که روی کل محور حقیقی تعریف شده‌اند (<code dir="ltr">normal</code> و <code dir="ltr">gumbel_right</code>) مشاهدات <code dir="ltr">ExactValue</code> و <code dir="ltr">RightCensoredValue</code> را می‌گیرند و مقدار می‌تواند منفی باشد. دادن جفت نوع دیگر <code dir="ltr">TypeError</code> ایجاد می‌کند. هر خانواده سانسور راست مستقل و <code dir="ltr">frequency_weights</code> را پشتیبانی می‌کند؛ <code dir="ltr">fit_weibull</code> گزینهٔ <code dir="ltr">fixed_shape</code> را هم می‌پذیرد.</p>

```python
from veridist import ExactLifetime, RightCensoredLifetime, fit

times = (120.0, 340.0, 560.0, 800.0, 1250.0, 1700.0)
observations = [ExactLifetime(t) for t in times] + [RightCensoredLifetime(2000.0)]
result = fit("weibull_min", observations)

shape, scale = result.parameters["shape"], result.parameters["scale"]
print(f"{result.family.value} shape={shape:.3f} scale={scale:.1f}")
print(result.observation_count, result.event_count, result.censored_count)
```

```text
weibull_min shape=1.216 scale=1154.2
7 6 1
```

<p dir="rtl" align="right">یک مشکل داده‌ای، مثلاً نمونه‌ای بدون رخداد مشاهده‌شده، به‌صورت شکست نوع‌دار برگردانده می‌شود و استثنا ایجاد نمی‌کند. نتیجهٔ موفق <code dir="ltr">family</code>، نگاشت فقط‌خواندنی <code dir="ltr">parameters</code> با نام‌های استاندارد پارامترها، <code dir="ltr">log_likelihood</code>، <code dir="ltr">observation_count</code>، <code dir="ltr">event_count</code>، <code dir="ltr">censored_count</code> و <code dir="ltr">converged</code> را همراه با ویژگی‌های مخصوص خانواده مانند <code dir="ltr">rate</code>، <code dir="ltr">shape</code>، <code dir="ltr">scale</code>، <code dir="ltr">mu</code> یا <code dir="ltr">sigma</code> در اختیار می‌گذارد. نتیجهٔ ناموفق <code dir="ltr">family</code>، <code dir="ltr">code</code> و شمارش‌ها را نشان می‌دهد. پروتکل‌های <code dir="ltr">FitSuccess</code> و <code dir="ltr">FitFailure</code> همین سطح مشترک را توصیف می‌کنند؛ بنابراین پیش از خواندن پارامترها بررسیِ مناسب <code dir="ltr">isinstance(result, FitSuccess)</code> است. برازش هرگز نقطه‌ای را که روی لبهٔ بازهٔ جست‌وجو قرار دارد به‌عنوان برآورد همگرا گزارش نمی‌کند؛ به‌جای آن شکستی مانند <code dir="ltr">BOUNDARY_SOLUTION</code> یا <code dir="ltr">DEGENERATE_SAMPLE</code> برمی‌گرداند.</p>

<h2 dir="rtl" align="right">ارزیابی توزیع‌ها روی اسکالرها و آرایه‌ها</h2>

<p dir="rtl" align="right"><code dir="ltr">logpdf</code>، <code dir="ltr">cdf</code>، <code dir="ltr">sf</code>، <code dir="ltr">ppf</code> و <code dir="ltr">sample</code> توزیعی را که از قبل مشخص کرده‌اید ارزیابی می‌کنند. خانواده اول می‌آید و پارامترها به‌صورت کلیدواژه پس از آن، برای مثال <code dir="ltr">cdf("gamma", 6.5, shape=5.0, scale=1.0)</code>. <code dir="ltr">logpdf</code> چگالی لگاریتمی، <code dir="ltr">sf</code> تابع بقا و <code dir="ltr">ppf</code> تابع چندک است؛ <code dir="ltr">sample(family, size, rng=rng, ...)</code> با یک مولد numpy که خودتان می‌دهید از خانواده نمونه می‌گیرد.</p>

<p dir="rtl" align="right">نقطه و هر پارامتر می‌تواند اسکالر پایتون یا numpy یا یک آرایه باشد و آن‌ها با یکدیگر هم‌پخش<sup id="fnref-broadcasting"><a href="#fn-broadcasting">۲۳</a></sup> می‌شوند. فراخوانی اسکالر یک <code dir="ltr">float</code> پایتون و هر ورودی آرایه‌ای یک آرایهٔ <code dir="ltr">float64</code> با شکل هم‌پخش‌شده برمی‌گرداند. <code dir="ltr">logpdf</code> بیرون از تکیه‌گاه برابر <code dir="ltr">-inf</code> است و <code dir="ltr">ppf</code> به احتمال‌هایی نیاز دارد که کاملاً بین ۰ و ۱ باشند. numpy فقط وقتی وارد می‌شود که آرایه بدهید.</p>

```python
import numpy as np

from veridist import cdf, logpdf, ppf

x = np.array([100.0, 200.0, 400.0])
print(np.round(cdf("weibull_min", x, shape=1.5, scale=500.0), 4))
print(round(logpdf("normal", 0.0, mu=0.0, sigma=1.0), 6))
print(round(ppf("gamma", 0.5, shape=2.0, scale=1.0), 6))
```

```text
[0.0856 0.2235 0.5111]
-0.918939
1.678347
```

<p dir="rtl" align="right">خانواده‌های نمایی، وایبول کمینه و گامبل راست از هسته‌های بومی numpy استفاده می‌کنند. خانواده‌های نرمال، لگ‌نرمال و گاما هسته‌های اسکالر راستی‌آزمایی‌شده را عنصر به عنصر ارزیابی می‌کنند؛ بنابراین مقدارهایی برابر با فراخوانی اسکالر می‌دهند، اما روی آرایه‌های بسیار بزرگ کندند. شکل قدیمی که پارامترها را به‌صورت نگاشت می‌دهد، <code dir="ltr">cdf("gamma", x, {"shape": 2.0, "scale": 1.0})</code>، هنوز کار می‌کند اما منسوخ شده است و در نسخهٔ ۳٫۰ حذف می‌شود.</p>

<h2 dir="rtl" align="right">ساخت مشاهدات از آرایه‌ها</h2>

<p dir="rtl" align="right"><code dir="ltr">lifetimes_from_arrays(time, event)</code> مشاهدات طول عمر را از دو ستون یک‌بعدی با طول برابر می‌سازد و <code dir="ltr">values_from_arrays(value, event)</code> همین کار را برای مقدارهای حقیقی انجام می‌دهد. <code dir="ltr">event</code> یک آرایهٔ بولی یا آرایهٔ صحیح شامل ۰ و ۱ است؛ درست یعنی رویداد مشاهده شده و نادرست یعنی مشاهده در آن زمان سانسور راست شده است. ستون‌ها ابتدا به‌صورت آرایهٔ کامل اعتبارسنجی می‌شوند و خطا نخستین سطر نامعتبر را نام می‌برد.</p>

```python
import numpy as np

from veridist import fit, lifetimes_from_arrays

time = np.array([120.0, 340.0, 560.0, 800.0, 2000.0])
event = np.array([1, 1, 1, 1, 0])
result = fit("exponential", lifetimes_from_arrays(time, event))
print(round(result.rate, 8))
```

```text
0.00104712
```

<h2 dir="rtl" align="right">گزارش عدم‌قطعیت یک برآورد</h2>

<p dir="rtl" align="right">هر برازش موفق یک متد <code dir="ltr">uncertainty()</code> دارد؛ آن را روی نتیجه فراخوانی کنید، زیرا ویژگی نیست. این متد یک <code dir="ltr">FitUncertainty</code> شامل <code dir="ltr">standard_errors</code> پارامترها، ماتریس کوواریانس<sup id="fnref-covariance"><a href="#fn-covariance">۲۴</a></sup> <code dir="ltr">covariance</code> به ترتیب استاندارد پارامترها و <code dir="ltr">confidence_intervals(level=0.95, method="wald")</code> برمی‌گرداند<sup id="fnref-confidence-interval"><a href="#fn-confidence-interval">۱۱</a></sup>. <code dir="ltr">method="wald"</code> برای پارامترهای مثبت بازه را در مقیاس لگاریتمی می‌سازد و بنابراین هرگز مقدار نامثبت ندارد. <code dir="ltr">method="profile"</code> درست‌نمایی نمایه‌ای<sup id="fnref-profile-likelihood"><a href="#fn-profile-likelihood">۲۵</a></sup> را وارونه می‌کند که برای نمونه‌های کوچک قابل‌اعتمادتر است؛ طرفی که هرگز قطع نمی‌شود <code dir="ltr">inf</code> (یا <code dir="ltr">0</code>) گزارش می‌شود. <code dir="ltr">method="exact"</code> بازهٔ کای‌دو را برای نرخ نمایی روی داده‌های بدون سانسور می‌دهد.</p>

<p dir="rtl" align="right">همان شیء کمیت‌های مشتق‌شده را همراه با بازه می‌دهد: <code dir="ltr">mean()</code> (میانگین زمان تا خرابی)، <code dir="ltr">quantile(p)</code> (یک عمر B<sup id="fnref-b-life"><a href="#fn-b-life">۲۶</a></sup>؛ <code dir="ltr">quantile(0.1)</code> همان B10 است) و <code dir="ltr">survival(t)</code>. هر کدام یک <code dir="ltr">DerivedEstimate</code> با <code dir="ltr">estimate</code>، <code dir="ltr">lower</code>، <code dir="ltr">upper</code>، <code dir="ltr">method</code> و <code dir="ltr">level</code> برمی‌گرداند. بازه‌های Wald از روش دلتا در مقیاسی استفاده می‌کنند که با دامنهٔ کمیت سازگار است و <code dir="ltr">method="profile"</code> برای خانواده‌های نمایی و وایبول در دسترس است.</p>

```python
from veridist import ExactLifetime, RightCensoredLifetime, fit

times = (120.0, 340.0, 560.0, 800.0, 1250.0, 1700.0)
observations = [ExactLifetime(t) for t in times] + [RightCensoredLifetime(2000.0)]
uncertainty = fit("weibull_min", observations).uncertainty()

intervals = uncertainty.confidence_intervals()
b10 = uncertainty.quantile(0.1)
print(tuple(round(value, 3) for value in intervals["shape"]))
print(round(b10.estimate, 1), round(b10.lower, 1), round(b10.upper, 1))
```

```text
(0.621, 2.382)
181.4 40.9 804.5
```

<p dir="rtl" align="right">وقتی اطلاعات مشاهده‌شده منفرد است، شکل Weibull ثابت شده یا نتیجه داده‌ای ندارد، <code dir="ltr">uncertainty()</code> به‌جای ایجاد استثنا یک مقدار <code dir="ltr">UncertaintyUnavailable</code> با <code dir="ltr">reason</code> پایدار برمی‌گرداند. پیش از خواندن بازه‌ها نوع را بررسی کنید.</p>

<h2 dir="rtl" align="right">ذخیرهٔ پیشرفت و ادامهٔ محاسبه</h2>

<p dir="rtl" align="right">برای داده‌ای که برنامهٔ شما از قبل به بخش‌های کوچک JSON تقسیم کرده است، تابع <code dir="ltr">fit_exponential_checkpointed_chunks</code> را از پکیج <code dir="ltr">veridist</code> وارد کنید. این تابع هر بخش را جداگانه پردازش می‌کند و آمار کافی<sup id="fnref-sufficient-statistics"><a href="#fn-sufficient-statistics">۱۹</a></sup> لازم برای ادامهٔ محاسبه را ذخیره می‌کند؛ نسخه‌ای از سطرهای اصلی داده را در فایل وضعیت نگه نمی‌دارد.</p>

<p dir="rtl" align="right">برای فایل CSV، تابع <code dir="ltr">veridist.execution.fit_exponential_checkpointed_csv</code> همین کار را همراه با امکان لغو انجام می‌دهد. شما فایل، تعریف ستون‌ها، محدودیت اندازه، محل ذخیرهٔ وضعیت، شناسهٔ نسخهٔ داده و در صورت نیاز تابع <code dir="ltr">cancel(cursor)</code> را می‌دهید. شناسهٔ نسخه یک برچسب دلخواه نیست: باید دقیقاً برابر با گوارش <bdi dir="ltr">SHA-256</bdi> فعلیِ فایل CSV باشد که پیش از خواندن هر سطر، با پیمایش جریانی فایل محاسبه می‌شود. اگر اجرا لغو شود، بخش کامل‌شده ابتدا ذخیره می‌شود؛ اجرای بعدی می‌تواند از آخرین سطر ثبت‌شده ادامه دهد.</p>

<p dir="rtl" align="right">فایل داده و شناسهٔ نسخهٔ آن باید بین دو اجرا ثابت بمانند: تغییر فایل، شناسهٔ منبع عمومیِ متفاوت، یا طرح‌وارهٔ ذخیره‌شدهٔ متفاوت، به‌جای ادامهٔ اجرا، یک عدم‌تطابق نوع‌دار برمی‌گرداند. برای ساخت ذخیرهٔ اولیهٔ یک فایل CSV، به‌جای ساختن دستیِ رکورد نقطهٔ بازرسی، از تابع <code dir="ltr">veridist.execution.create_checkpointed_csv_store</code> استفاده کنید. سازوکار ذخیرهٔ وضعیت<sup id="fnref-checkpoint"><a href="#fn-checkpoint">۲۰</a></sup> برای ادامهٔ محاسبه روی همان رایانه طراحی شده است. پیاده‌سازی فعلی از SQLite محلی استفاده می‌کند، اما سطرهای خام CSV را در آن کپی نمی‌کند. این فایل به‌صورت خودکار رمزگذاری یا احراز هویت نمی‌شود؛ بنابراین باید آن را مانند سایر فایل‌های کاری در محل امن نگه دارید. <a href="../../examples/checkpoint_resume.py">مثال ذخیره و ادامهٔ اجرا</a> روند دو مرحله‌ای را نشان می‌دهد.</p>

<h2 dir="rtl" align="right">ابزارهای سطح پایین<sup id="fnref-low-level-tools"><a href="#fn-low-level-tools">۲۱</a></sup> برای داده‌های آماده</h2>

<p dir="rtl" align="right">اگر داده را خود برنامهٔ شما تولید یا بخش‌بندی می‌کند، <code dir="ltr">IterableDataSource</code> آن بخش‌ها را همراه با <code dir="ltr">DataSourceMetadata</code> تغییرناپذیر و اعلام صریح <code dir="ltr">Replayability</code> دریافت می‌کند. در حالت <code dir="ltr">SINGLE_PASS</code> داده فقط یک‌بار خوانده می‌شود. در حالت <code dir="ltr">REPLAYABLE</code> باید تابعی بدهید که هر بار یک پیمایش تازه از داده بسازد. حالت <code dir="ltr">CHECKPOINT_REPLAYABLE</code> در این adapter هنوز پیاده نشده است؛ استفاده از آن خطای <code dir="ltr">CHECKPOINT_REQUIRED</code> می‌دهد.</p>

<p dir="rtl" align="right"><code dir="ltr">FAMILY_REGISTRY</code> و <code dir="ltr">FamilyId</code> مشخصات شش خانوادهٔ آماری ارزیابی‌شده را نگه می‌دارند. <code dir="ltr">evaluate_log_density</code> چگالی لگاریتمی یک مقدار را با پارامترهای داده‌شده محاسبه می‌کند. <code dir="ltr">reduce_log_likelihood_chunks</code> همین محاسبه را برای بخش‌های متعدد داده جمع می‌کند و نتیجه‌ای مستقل از نحوهٔ بخش‌بندی می‌سازد. این توابع پارامترهای مدل را تخمین نمی‌زنند و توزیع‌ها را رتبه‌بندی نمی‌کنند. قرارداد دقیق آن‌ها در <a href="families-log-density-likelihood.md">راهنمای خانواده‌ها و درست‌نمایی لگاریتمی</a> آمده است.</p>

<p dir="rtl" align="right">برای داده‌های سانسورشده، <code dir="ltr">reduce_lifetime_log_likelihood_chunks</code> (خانواده‌های <code dir="ltr">exponential</code>، <code dir="ltr">weibull_min</code>، <code dir="ltr">lognormal</code> و <code dir="ltr">gamma</code> با <code dir="ltr">ExactLifetime</code> و <code dir="ltr">RightCensoredLifetime</code>) و <code dir="ltr">reduce_value_log_likelihood_chunks</code> (خانواده‌های <code dir="ltr">normal</code> و <code dir="ltr">gumbel_right</code> با <code dir="ltr">ExactValue</code> و <code dir="ltr">RightCensoredValue</code>) چگالی لگاریتمی هر مشاهدهٔ دقیق و لگاریتم بقای هر مشاهدهٔ سانسورشده از راست را جمع می‌کنند. آن‌ها یک <code dir="ltr">FamilyId</code> و پارامترهای استاندارد را می‌گیرند، دقیقاً مانند <code dir="ltr">reduce_log_likelihood_chunks</code> انباشته می‌کنند و در پارامترهای یک برازش، <code dir="ltr">log_likelihood</code> همان برازش را بازتولید می‌کنند.</p>

```python
from veridist import (
    ExactLifetime,
    FamilyId,
    RightCensoredLifetime,
    reduce_lifetime_log_likelihood_chunks,
)

chunks = [[ExactLifetime(120.0), ExactLifetime(340.0)], [RightCensoredLifetime(2000.0)]]
result = reduce_lifetime_log_likelihood_chunks(
    FamilyId.WEIBULL_MIN, chunks, shape=1.2, scale=1100.0
)
print(result.observation_count, round(result.total_log_likelihood, 6))
```

```text
3 -16.682975
```

<h2 dir="rtl" align="right">آنچه بستهٔ سطح بالا صادر می‌کند</h2>

<p dir="rtl" align="right">API عمومی را از <code dir="ltr">veridist</code> وارد کنید: شش تابع برازش و <code dir="ltr">fit</code>؛ <code dir="ltr">FamilyId</code>، <code dir="ltr">logpdf</code>، <code dir="ltr">cdf</code>، <code dir="ltr">sf</code>، <code dir="ltr">ppf</code> و <code dir="ltr">sample</code>؛ انواع مشاهده همراه با <code dir="ltr">lifetimes_from_arrays</code> و <code dir="ltr">values_from_arrays</code>؛ پروتکل‌های <code dir="ltr">FitSuccess</code> و <code dir="ltr">FitFailure</code>؛ نقطه‌های ورود CSV و نقطهٔ بازرسی از جمله <code dir="ltr">create_checkpointed_csv_store</code> و <code dir="ltr">fit_exponential_checkpointed_csv</code>؛ سه کاهندهٔ درست‌نمایی لگاریتمی؛ و کلاس‌های خطای <code dir="ltr"><bdi dir="ltr">Veridist</bdi>Error</code>، <code dir="ltr">CapabilityError</code> و <code dir="ltr">EngineContractError</code>. ذخیره‌گاه‌ها، بافرها، منشأ و انواع خروجی در <code dir="ltr">veridist.engine</code> می‌مانند که <code dir="ltr">CapabilityError</code> را هم صادر می‌کند.</p>

<h2 dir="rtl" align="right">ارتقا از نسخهٔ ۱٫۰</h2>

<p dir="rtl" align="right">کد نسخهٔ ۱٫۰ همچنان کار می‌کند. دو شکل منسوخ شده‌اند و در نسخهٔ ۳٫۰ حذف می‌شوند: شکل نگاشتیِ <code dir="ltr">cdf</code>، <code dir="ltr">sf</code>، <code dir="ltr">ppf</code> و <code dir="ltr">sample</code>، و بخش‌های <code dir="ltr">bytes</code> خام برای <code dir="ltr">fit_exponential_checkpointed_chunks</code>. <a href="../migration-2.0.md">راهنمای مهاجرت</a> همهٔ تغییرها را فهرست می‌کند، از جمله تغییر مجوز به Business Source License 1.1 با اجازهٔ غیرتجاری.</p>

<details dir="rtl" align="right">
<summary>جزئیات فنی و محدودیت‌های نسخهٔ فعلی</summary>

<p dir="rtl" align="right">خواندن فایل CSV در یک پیمایش انجام می‌شود. محدودیت‌های اندازه فقط حجم بخش‌های داده‌ای را که خود adapter نگه می‌دارد کنترل می‌کنند و سقف کل حافظهٔ فرایند یا سرعت اجرا نیستند. پشتیبانی عمومی از همهٔ منابع داده، اجرای توزیع‌شده و بازیابی بین چند رایانه در نسخهٔ فعلی وجود ندارد؛ موارد برنامه‌ریزی‌شده و مرزهای دقیق در <a href="../../KNOWN_LIMITS.fa.md">محدودیت‌های شناخته‌شده</a> ثبت شده‌اند. موفق‌بودن محاسبه نیز به‌تنهایی ثابت نمی‌کند که توزیع نمایی برای داده مناسب است.</p>

</details>

<h2 dir="rtl" align="right">اصطلاحات این صفحه</h2>

<p id="fn-fitting" dir="rtl" align="right"><strong>۱.</strong> <bdi dir="ltr">Distribution fitting</bdi> — برآورد پارامترهای یک توزیع از روی داده و بررسی سازگاری آن با مشاهدات. <a href="#fnref-fitting" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-scalar" dir="rtl" align="right"><strong>۲.</strong> <bdi dir="ltr">Scalar operation</bdi> — عملیاتی روی یک مقدار عددی؛ همین عملیات آرایه‌ها را هم می‌پذیرند. <a href="#fnref-scalar" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-caller" dir="rtl" align="right"><strong>۳.</strong> <bdi dir="ltr">Caller</bdi> — کد یا برنامه‌ای که تابع کتابخانه را صدا می‌زند و ورودی‌هایش را فراهم می‌کند. <a href="#fnref-caller" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-distributed-recovery" dir="rtl" align="right"><strong>۴.</strong> <bdi dir="ltr">Distributed recovery</bdi> — ادامه‌دادن یک محاسبه روی رایانه یا سرویس دیگری با وضعیت مشترک. <a href="#fnref-distributed-recovery" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-rate" dir="rtl" align="right"><strong>۵.</strong> <bdi dir="ltr">Rate</bdi> — تعداد مورد انتظار رخداد در هر واحد زمان؛ نرخ با احتمال رخداد یکسان نیست. <a href="#fnref-rate" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-right-censoring" dir="rtl" align="right"><strong>۶.</strong> <bdi dir="ltr">Independent right censoring</bdi> — تا پایان مشاهده رخداد دیده نشده و فرض می‌شود علت پایان مشاهده از زمان رخداد مستقل است. <a href="#fnref-right-censoring" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-provenance" dir="rtl" align="right"><strong>۷.</strong> <bdi dir="ltr">Provenance</bdi> — اطلاعات ثبت‌شده دربارهٔ منبع داده و نحوهٔ اجرای محاسبه برای بررسی بعدی نتیجه. <a href="#fnref-provenance" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-byte-limits" dir="rtl" align="right"><strong>۸.</strong> <bdi dir="ltr">Byte limits</bdi> — سقف حجم داده‌ای که هر بخش و صف پردازش می‌توانند هم‌زمان نگه دارند؛ این مقدار سقف کل RAM برنامه نیست. <a href="#fnref-byte-limits" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-typed-non-estimate" dir="rtl" align="right"><strong>۹.</strong> <bdi dir="ltr">Typed statistical non-estimate</bdi> — نتیجه‌ای ساخت‌یافته که می‌گوید محاسبه اجرا شده اما داده اجازهٔ برآورد معتبر را نداده است و دلیل را مشخص می‌کند. <a href="#fnref-typed-non-estimate" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-typed-outcome" dir="rtl" align="right"><strong>۱۰.</strong> <bdi dir="ltr">Typed outcome</bdi> — نتیجه‌ای با نوع و کد مشخص که برنامه می‌تواند بدون تحلیل متن آزاد آن را بررسی کند. <a href="#fnref-typed-outcome" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-confidence-interval" dir="rtl" align="right"><strong>۱۱.</strong> <bdi dir="ltr">Confidence interval</bdi> — بازه‌ای که عدم‌قطعیت برآورد پارامتر را با یک روش آماری مشخص نشان می‌دهد. <a href="#fnref-confidence-interval" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-goodness-of-fit" dir="rtl" align="right"><strong>۱۲.</strong> <bdi dir="ltr">Goodness-of-fit test</bdi> — بررسی آماری میزان سازگاری شکل توزیع انتخاب‌شده با داده. <a href="#fnref-goodness-of-fit" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-weight" dir="rtl" align="right"><strong>۱۳.</strong> <bdi dir="ltr">Weight</bdi> — عددی که سهم یک مشاهده را در محاسبه بیشتر یا کمتر می‌کند. <a href="#fnref-weight" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-covariate" dir="rtl" align="right"><strong>۱۴.</strong> <bdi dir="ltr">Covariate</bdi> — متغیری مانند دما یا فشار که ممکن است روی زمان رخداد اثر بگذارد. <a href="#fnref-covariate" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-truncation" dir="rtl" align="right"><strong>۱۵.</strong> <bdi dir="ltr">Truncation</bdi> — حذف‌شدن بخشی از جامعه از نمونه به‌علت سازوکار ورود یا مشاهده، نه صرفاً نامشخص‌بودن زمان رخداد. <a href="#fnref-truncation" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-left-censoring" dir="rtl" align="right"><strong>۱۶.</strong> <bdi dir="ltr">Left censoring</bdi> — فقط می‌دانیم رخداد پیش از یک زمان مشخص اتفاق افتاده است. <a href="#fnref-left-censoring" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-interval-censoring" dir="rtl" align="right"><strong>۱۷.</strong> <bdi dir="ltr">Interval censoring</bdi> — فقط می‌دانیم رخداد در فاصلهٔ بین دو زمان اتفاق افتاده است. <a href="#fnref-interval-censoring" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-free-location" dir="rtl" align="right"><strong>۱۸.</strong> <bdi dir="ltr">Free location parameter</bdi> — پارامتر جابه‌جایی توزیع که به‌جای ثابت‌بودن، از داده برآورد می‌شود. <a href="#fnref-free-location" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-sufficient-statistics" dir="rtl" align="right"><strong>۱۹.</strong> <bdi dir="ltr">Sufficient statistics</bdi> — خلاصه‌های عددی لازم برای برآورد پارامتر که در این محاسبه جای نگهداری همهٔ سطرهای خام را می‌گیرند. <a href="#fnref-sufficient-statistics" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-checkpoint" dir="rtl" align="right"><strong>۲۰.</strong> <bdi dir="ltr">Checkpoint</bdi> — وضعیت میانی ذخیره‌شده که اجازه می‌دهد اجرای سازگار از آخرین بخش ثبت‌شده ادامه پیدا کند. <a href="#fnref-checkpoint" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-low-level-tools" dir="rtl" align="right"><strong>۲۱.</strong> <bdi dir="ltr">Low-level tools</bdi> — توابع پایه‌ای‌تر برای زمانی که برنامهٔ شما آماده‌سازی داده، بخش‌بندی داده یا انتخاب پارامترها را خودش انجام می‌دهد. این ابزارها معمولاً برای مسیر شروع سریع کاربر نهایی نیستند. <a href="#fnref-low-level-tools" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-uncertainty" dir="rtl" align="right"><strong>۲۲.</strong> <bdi dir="ltr">Uncertainty</bdi> — میزان فاصلهٔ احتمالیِ برآوردی که از نمونه‌ای محدود به‌دست آمده با مقدار واقعی؛ معمولاً با خطاهای معیار و فاصله‌های اطمینان گزارش می‌شود. <a href="#fnref-uncertainty" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-broadcasting" dir="rtl" align="right"><strong>۲۳.</strong> <bdi dir="ltr">Broadcasting</bdi> — قاعدهٔ numpy برای ترکیب عنصر به عنصر آرایه‌هایی با شکل‌های متفاوت اما سازگار. <a href="#fnref-broadcasting" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-covariance" dir="rtl" align="right"><strong>۲۴.</strong> <bdi dir="ltr">Covariance matrix</bdi> — جدولی از واریانس برآوردهای پارامترها و کوواریانس هر جفت از آن‌ها؛ ریشهٔ دوم قطر آن خطاهای معیار است. <a href="#fnref-covariance" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-profile-likelihood" dir="rtl" align="right"><strong>۲۵.</strong> <bdi dir="ltr">Profile likelihood</bdi> — درست‌نمایی بیشینه‌شده روی پارامترهای دیگر به ازای هر مقدار از پارامتر مورد نظر؛ بازه‌های ساخته‌شده از آن شکل واقعی درست‌نمایی را دنبال می‌کنند. <a href="#fnref-profile-likelihood" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-b-life" dir="rtl" align="right"><strong>۲۶.</strong> <bdi dir="ltr">B-life</bdi> — زمانی که تا آن درصد معینی از واحدها خراب شده‌اند؛ B10 زمانی است که تا آن ۱۰ درصد خراب شده‌اند. <a href="#fnref-b-life" aria-label="بازگشت به متن">↩</a></p>
